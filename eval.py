import argparse
import torch
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    encoder.load_state_dict(torch.load(args.encoder))
    decoder.load_state_dict(torch.load(args.decoder))
    encoder.eval()
    decoder.eval()

    # Train the models
    total_step = len(data_loader)
    with torch.no_grad():
        total = 0
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)

            features = encoder(images)
            outputs = decoder.sample(features)

            # Remove padding from captions
            captions = captions.cpu().data.tolist()
            for idx, caption in enumerate(captions):
                # 1 corresponds to <start>, 2 corresponds to <end>
                captions[idx] = caption[caption.index(1): caption.index(2)]
                captions[idx] = [vocab.idx2word[i] for i in captions[idx]]

            # Remove start and end from the output
            outputs = outputs.cpu().data.tolist()
            for idx, output in enumerate(outputs):
                hypothesis = []
                for idx_word in output:
                    if idx_word == 1:
                        hypothesis = []
                    elif idx_word == 2:
                        break
                    else:
                        hypothesis.append(vocab.idx2word[idx_word])

                score = sentence_bleu([captions[idx]], hypothesis)
                total = total + score
                print("Actual", captions[idx])
                print("Hypothesis", hypothesis)
                print(score)

            # Print log info
            if i % args.log_step == 0:
                print('Step [{}/{}], Avg BLEU: {:.4f}'
                      .format(i+1, total_step, total/(i+1)*data_loader.batch_size))

        avg = total/len(data_loader.dataset)
        print(avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, required=True, help='path for loading trained models')
    parser.add_argument('--decoder', type=str, required=True, help='path for loading trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/valresized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_val2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers in lstm')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
    main(args)
