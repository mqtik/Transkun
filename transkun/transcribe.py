import argparse
import importlib.resources

from .Data import writeMidi
import torch
import moduleconf
import numpy as np
from .Util import computeParamSize


_cached_model = None
_cached_device = None


def readAudio(path, normalize=True):
    import subprocess
    import tempfile
    import os
    import soundfile as sf

    ext = os.path.splitext(path)[1].lower()
    if ext in ('.wav', '.flac', '.ogg'):
        y, fs = sf.read(path, dtype='float32')
    else:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', path, '-ar', '44100', '-ac', '2', tmp.name],
                check=True, capture_output=True,
            )
            y, fs = sf.read(tmp.name, dtype='float32')
        finally:
            os.unlink(tmp.name)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if not normalize:
        y = (y * 2**15).astype(np.int16)
    return fs, y


def load_model(device="cpu", weight=None, conf=None):
    global _cached_model, _cached_device

    if _cached_model is not None and _cached_device == device:
        return _cached_model

    pretrained = importlib.resources.files(__package__).joinpath("pretrained")
    if weight is None:
        weight = str(pretrained.joinpath("2.0.pt"))
    if conf is None:
        conf = str(pretrained.joinpath("2.0.conf"))

    confManager = moduleconf.parseFromFile(conf)
    TransKun = confManager["Model"].module.TransKun
    model_conf = confManager["Model"].config

    checkpoint = torch.load(weight, map_location=device)
    model = TransKun(conf=model_conf).to(device)

    if "best_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["best_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.eval()
    _cached_model = model
    _cached_device = device
    return model


def transcribe_audio(audio_path, device="cpu", weight=None, conf=None):
    model = load_model(device=device, weight=weight, conf=conf)

    with torch.no_grad():
        fs, audio = readAudio(audio_path)

        if fs != model.fs:
            import soxr
            audio = soxr.resample(audio, fs, model.fs)

        x = torch.from_numpy(audio).to(device)
        notes = model.transcribe(x, discardSecondHalf=False)

    return writeMidi(notes)


def main():
    pretrained = importlib.resources.files(__package__).joinpath("pretrained")
    defaultWeight = str(pretrained.joinpath("2.0.pt"))
    defaultConf = str(pretrained.joinpath("2.0.conf"))

    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("audioPath", help="path to the input audio file")
    argumentParser.add_argument("outPath", help="path to the output MIDI file")
    argumentParser.add_argument("--weight", default=defaultWeight, help="path to the pretrained weight")
    argumentParser.add_argument("--conf", default=defaultConf, help="path to the model conf")
    argumentParser.add_argument("--device", default="cpu", nargs="?", help="The device used to perform the most computations (optional), DEFAULT: cpu")
    argumentParser.add_argument("--segmentHopSize", type=float, required=False, help="The segment hopsize for processing the entire audio file (s)")
    argumentParser.add_argument("--segmentSize", type=float, required=False, help="The segment size for processing the entire audio file (s)")

    args = argumentParser.parse_args()

    model = load_model(device=args.device, weight=args.weight, conf=args.conf)

    with torch.no_grad():
        fs, audio = readAudio(args.audioPath)

        if fs != model.fs:
            import soxr
            audio = soxr.resample(audio, fs, model.fs)

        x = torch.from_numpy(audio).to(args.device)
        notesEst = model.transcribe(x, stepInSecond=args.segmentHopSize, segmentSizeInSecond=args.segmentSize, discardSecondHalf=False)

    outputMidi = writeMidi(notesEst)
    outputMidi.write(args.outPath)


if __name__ == "__main__":
    main()
