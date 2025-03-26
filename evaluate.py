import argparse
import os
import pickle
from glob import glob

import PIL.Image
import numpy as np
import torch
from diffusers.utils import load_image
from torch.utils import data

from tqdm import tqdm

from psc import Latent_PSC_compress, Latent_PSC_decompress, Latent_PSC_simulate, set_seed

class ImageDataset(data.Dataset):
    def __init__(self, data_dir):
        self.imgs = sorted(glob(f"{data_dir}/*.png"), key=lambda x: int(os.path.basename(x).split(".")[0]))

    def __getitem__(self, index):
        img = torch.from_numpy(np.array(load_image(self.imgs[index]))).permute(2, 0, 1)
        return img

    def __len__(self):
        return len(self.imgs)


def main(data_dir, outdir, use_captions, num_samples, rank, iters, start, end, num_variations, simulate=False):
    dataset = ImageDataset(data_dir)
    if start is not None and end is not None:
        dataset = torch.utils.data.Subset(dataset, range(start, end))

    avg_psnr, bpp = 0, 0
    set_seed(0)
    for i, image in enumerate(tqdm(dataset, desc="Directory")):
        im = image / 127.5 - 1
        prompt, text_bpp = "", 0.0
        if use_captions:
            from text_utils import caption, compress_text
            prompt = caption(image)
            byte_stream_text = compress_text(prompt)
            text_bpp = len(byte_stream_text) * 8 / image.shape[-2] ** 2
        if simulate:
            set_seed(i)
            byte_stream, decompressed = Latent_PSC_simulate(im, prompt, iters, rank, num_samples, num_restorations=num_variations)
        else:
            set_seed(i)
            byte_stream = Latent_PSC_compress(im, prompt, iters, rank, num_samples)
            set_seed(i)
            decompressed = Latent_PSC_decompress(byte_stream, prompt, iters, rank, num_samples, num_restorations=num_variations)
        assert len(byte_stream) == iters * rank
        decompressed = decompressed * 127.5 + 127.5
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            numpy_im = decompressed.clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            name = os.path.join(outdir, f"sample_{i + (start if start is not None else 0)}")
            PIL.Image.fromarray(numpy_im).save(name + ".png")
            torch.save(byte_stream, name + ".pth")
            if use_captions:
                with open(name + "_txt.pkl", "wb") as thefile:
                    pickle.dump(byte_stream_text, thefile)
        psnr = 10 * torch.log10(255**2 / (image.to(decompressed).float() - decompressed.float()).pow(2).mean())
        avg_psnr = avg_psnr + (psnr.float() / len(dataset)).cpu().item()
        bpp = bpp + ((text_bpp + (len(byte_stream) * 8 / image.shape[-2] ** 2)) / len(dataset))
    print(f"PSNR: {avg_psnr}, BPP: {bpp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('-c' '--use-captions', action="store_true")
    parser.add_argument('-v', '--num-variations', type=int, default=1)
    parser.add_argument('-n', '--num-samples', type=int, default=None)
    parser.add_argument('-r', '--rank', type=int, default=12)
    parser.add_argument('-i', '--iters', type=int, default=256)
    parser.add_argument('-s', '--simulate', action="store_true")
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    main(**vars(args))
