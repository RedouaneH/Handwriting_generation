"""
generate_handwriting.py

Wrapper around the provided Hand class to generate handwriting SVGs from plain text.
Provides a single function `generate_handwriting(...)` and a small CLI.

Dependencies: numpy, svgwrite, tensorflow (as required by the rnn.restore), wand, pillow,
and the local modules `drawing`, `lyrics`, and `rnn` (same project layout as your original file). 
Also expects a `styles/` directory if you use style priming.

Usage examples (from CLI):
    python generate_handwriting.py --text "Hello world" --output out.svg --style 9 --bias 0.75 --stroke_width 2
    python generate_handwriting.py --text "Hello world" --output out.png --format png --style 9 --bias 0.75 --stroke_width 2

"""
import os
import argparse
import textwrap
from typing import List, Optional, Union
import matplotlib.pyplot as plt

# keep the same imports as the original implementation
import logging
import numpy as np
import svgwrite

# New imports for PNG conversion
from wand.image import Image as WandImage
from wand.color import Color
from PIL import Image

import drawing
import lyrics
from rnn import rnn


def svg_to_cropped_transparent_png(svg_path, out_path,
                                   pad=8,   # marge en pixels autour du tracé
                                   white_tol=0.98,  # seuil de "blanc" (0..1)
                                   method='saturation'  # 'luminance', 'distance', 'saturation'
                                  ):
    """Convert SVG to cropped transparent PNG with improved color handling.
    
    Uses advanced color detection methods to better handle colored text (not just black).
    The 'saturation' method works best for colorful text like yellow, cyan, etc.
    """
    # 1) SVG -> PNG (temp)
    with WandImage(filename=svg_path, format="svg") as img:
        img.format = 'png32'                          # RGBA
        img.background_color = Color("transparent")
        img.alpha_channel = 'activate'
        img.save(filename="temp.png")
    
    # 2) Ouvrir avec PIL, convertir en tableau float normalisé (0..1)
    im = Image.open("temp.png").convert("RGBA")
    arr = np.asarray(im).astype(np.float32) / 255.0  # shape (H,W,4)
    rgb = arr[..., :3]   # R,G,B
    
    # 3) Méthodes améliorées pour calculer la "blancheur"
    if method == 'luminance':
        # Luminance perceptuelle (pondérée selon vision humaine)
        # L'œil humain est plus sensible au vert, puis rouge, puis bleu
        brightness = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        
    elif method == 'distance':
        # Distance euclidienne au blanc pur
        white = np.array([1.0, 1.0, 1.0])
        distance_to_white = np.sqrt(np.sum((rgb - white)**2, axis=2))
        # Inverser : distance faible = proche du blanc = brightness élevée
        max_distance = np.sqrt(3.0)  # distance max possible dans l'espace RGB
        brightness = 1.0 - (distance_to_white / max_distance)
        
    elif method == 'saturation':
        # Combinaison luminance + saturation (MEILLEUR pour les couleurs vives)
        # Un pixel blanc a une forte luminance ET une faible saturation
        luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        
        # Calcul de la saturation (écart entre canal max et min)
        rgb_max = np.max(rgb, axis=2)
        rgb_min = np.min(rgb, axis=2)
        saturation = np.where(rgb_max > 0, (rgb_max - rgb_min) / rgb_max, 0)
        
        # Un pixel blanc a une luminance élevée ET une saturation faible
        # On pondère : plus c'est saturé, moins c'est "blanc"
        brightness = luminance * (1.0 - saturation * 0.8)
        
    else:  # fallback à l'ancienne méthode
        brightness = rgb.mean(axis=2)
    
    # 4) Calcul de l'alpha basé sur la brightness
    alpha = 1.0 - brightness       # plus c'est sombre, plus alpha proche de 1
    # Saturer les faibles alpha en dessous d'un seuil blanc_tol
    alpha = np.clip((alpha - (1.0 - white_tol)) / white_tol, 0.0, 1.0)
    # maintenant alpha est entre 0 et 1, 0=blanc, 1=plein trait
    
    # 5) Récupérer la couleur du trait (undo white premultiplication)
    #    o = fg*alpha + 1*(1-alpha)  =>  fg = (o - (1-alpha)) / alpha
    o = rgb
    eps = 1e-6
    alpha_safe = np.where(alpha > eps, alpha, 1.0)  # éviter division par 0
    fg = (o - (1.0 - alpha[..., None])) / alpha_safe[..., None]
    fg = np.clip(fg, 0.0, 1.0)
    # Si alpha très petit (presque transparent), mettre rgb à 0 pour éviter artefacts
    mask_transparent = (alpha <= eps)
    fg[mask_transparent, :] = 0.0
    
    # 6) Construire image RGBA finale (0..255 uint8)
    out_arr = np.dstack(( (fg * 255).astype(np.uint8),
                          (alpha * 255).astype(np.uint8) ))
    final_im = Image.fromarray(out_arr, mode="RGBA")
    
    # 7) Crop sur le contenu non transparent puis ajouter padding
    bbox = final_im.getbbox()
    if bbox is None:
        # image vide ? on sauvegarde l'originale transparente
        final_im.save(out_path)
        return
    
    left, upper, right, lower = bbox
    left = max(0, left - pad)
    upper = max(0, upper - pad)
    right = min(final_im.width, right + pad)
    lower = min(final_im.height, lower + pad)
    cropped = final_im.crop((left, upper, right, lower))
    
    # 8) Sauvegarder
    cropped.save(out_path)
    
    # 9) Nettoyer le fichier temporaire
    if os.path.exists("temp.png"):
        os.remove("temp.png")


class Hand(object):

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # reduce noisy logging from TensorFlow / model code
        logging.getLogger().setLevel(logging.WARNING)

        self.nn = rnn(
            log_dir='logs',
            checkpoint_dir='checkpoints',
            prediction_dir='predictions',
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(self, filename: str, lines: List[str], biases: Optional[List[float]] = None,
              styles: Optional[List[int]] = None, stroke_colors: Optional[List[str]] = None,
              stroke_widths: Optional[List[Union[int, float]]] = None):
        valid_char_set = set(drawing.alphabet)
        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    (
                        "Each line must be at most 75 characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )

        strokes = self._sample(lines, biases=biases, styles=styles)
        self._draw(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40 * max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5] * num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load('styles/style-{}-strokes.npy'.format(style))
                c_p = np.load('styles/style-{}-chars.npy'.format(style)).tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

    def _draw(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None):
        stroke_colors = stroke_colors or ['black'] * len(lines)
        stroke_widths = stroke_widths or [2] * len(lines)

        line_height = 60
        view_width = 1000
        view_height = line_height * (len(strokes) + 1)

        dwg = svgwrite.Drawing(filename=filename)
        dwg.viewbox(width=view_width, height=view_height)
        dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

        initial_coord = np.array([0, -(3 * line_height / 4)])
        for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):

            if not line:
                initial_coord[1] -= line_height
                continue

            offsets[:, :2] *= 1.5
            strokes = drawing.offsets_to_coords(offsets)
            strokes = drawing.denoise(strokes)
            strokes[:, :2] = drawing.align(strokes[:, :2])

            strokes[:, 1] *= -1
            strokes[:, :2] -= strokes[:, :2].min() + initial_coord
            strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*strokes.T):
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
            path = svgwrite.path.Path(p)
            path = path.stroke(color=color, width=width, linecap='round').fill("none")
            dwg.add(path)

            initial_coord[1] -= line_height

        dwg.save()


# single global Hand instance to avoid re-loading model multiple times
_HAND: Optional[Hand] = None


def _get_hand() -> Hand:
    global _HAND
    if _HAND is None:
        _HAND = Hand()
    return _HAND


def _ensure_list(v, length, name: str, cast_type=None):
    """Convert v to a list of length `length`.
    If v is None -> return None.
    If v is a single value -> repeat it.
    If v is a list shorter/longer -> pad with last value or truncate.
    """
    if v is None:
        return None
    if not isinstance(v, (list, tuple, np.ndarray)):
        v = [v]
    v = list(v)
    if cast_type is not None:
        v = [cast_type(x) for x in v]
    if len(v) < length:
        v = v + [v[-1]] * (length - len(v))
    elif len(v) > length:
        v = v[:length]
    return v


def generate_handwriting(
    text: str,
    output_path: str = 'img/output.png',
    width: int = 75,
    bias: Optional[Union[float, List[float]]] = 0.5,
    style: Optional[Union[int, List[int]]] = 4,
    color: Union[str, List[str]] = 'black',
    stroke_width: Union[int, float, List[Union[int, float]]] = 2,
    output_format: str = 'png',
    transparency_method: str = 'distance'
) -> str:
    """Generate handwritten text in SVG or PNG format.

    Parameters
    ----------
    text: input text. Newlines supported. Long lines are wrapped at width.
    output_path: output file path. Extension should match output_format.
    width: maximum characters per line (default: 75).
    bias: float or list of floats, model sampling bias (higher -> more deterministic).
    style: None or an int or list of ints (style priming indices).
    color: single color or list of colors per line.
    stroke_width: single width or list of widths per line.
    output_format: 'svg' or 'png' (default: 'svg').
    transparency_method: method for PNG background removal - 'saturation' (best for colors), 
                        'luminance' (perceptual), 'distance' (mathematical) (default: 'saturation').

    Returns
    -------
    output_path (str) of the saved file.
    """
    import unicodedata

    # Validate output format
    if output_format.lower() not in ['svg', 'png']:
        raise ValueError("output_format must be 'svg' or 'png'")

    # normalize input text and wrap lines
    raw_lines = text.splitlines() if isinstance(text, str) else [str(text)]
    lines: List[str] = []
    for L in raw_lines:
        if L.strip() == '':
            lines.append('')
        else:
            wrapped = textwrap.wrap(L, width=width)
            if not wrapped:
                lines.append('')
            else:
                lines.extend(wrapped)

    if not lines:
        lines = ['']

    # sanitize lines: remove accents and chars not in drawing.alphabet
    valid_char_set = set(drawing.alphabet)
    sanitized_lines: List[str] = []
    removed_summary = {}

    for line in lines:
        # decompose accents and drop non-ascii (é -> e)
        norm = unicodedata.normalize('NFKD', line)
        ascii_line = norm.encode('ascii', 'ignore').decode('ascii')

        # filter out characters not supported by the model
        filtered = ''.join(ch for ch in ascii_line if ch in valid_char_set)

        # track removed characters for warning
        removed_from_original = set(line) - set(ascii_line)
        removed_by_filter = set(ascii_line) - set(filtered)
        removed = removed_from_original.union(removed_by_filter)
        if removed:
            removed_summary[line] = ''.join(sorted(removed))

        sanitized_lines.append(filtered)

    if removed_summary:
        # small, user-friendly warning listing lines that changed
        logging.warning("Some input characters were removed or replaced because they are not supported by the model. Example removals: %s",
                        {k: removed_summary[k] for k in list(removed_summary)[:5]})

    # use sanitized lines for generation
    lines = sanitized_lines

    # prepare style and bias lists
    styles = None
    if style is not None:
        if isinstance(style, int):
            styles = [style] * len(lines)
        else:
            styles = list(style)
            styles = _ensure_list(styles, len(lines), 'style', int)

    biases = _ensure_list(bias, len(lines), 'bias', float)
    stroke_colors = _ensure_list(color, len(lines), 'stroke_color', str)
    stroke_widths = _ensure_list(stroke_width, len(lines), 'stroke_width', float)

    # ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Generate SVG first
    if output_format.lower() == 'png':
        # For PNG output, generate SVG to temp file first
        temp_svg = output_path.rsplit('.', 1)[0] + '_temp.svg'
        svg_path = temp_svg
    else:
        svg_path = output_path

    hand = _get_hand()
    hand.write(filename=svg_path, lines=lines, biases=biases, styles=styles,
               stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    # Convert to PNG if requested using improved method
    if output_format.lower() == 'png':
        svg_to_cropped_transparent_png(svg_path, output_path, method=transparency_method)
        # Remove temp SVG file
        if os.path.exists(temp_svg):
            os.remove(temp_svg)

    return output_path


def _parse_list_arg(s: Optional[str]):
    if s is None:
        return None
    if ',' in s:
        items = [item.strip() for item in s.split(',') if item.strip()]
        # try to convert to int if possible, else float, else leave string
        parsed = []
        for it in items:
            try:
                parsed.append(int(it))
                continue
            except Exception:
                pass
            try:
                parsed.append(float(it))
                continue
            except Exception:
                parsed.append(it)
        return parsed
    # single value - try int/float
    try:
        return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        pass
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate handwriting SVG or PNG from text')
    
    # Text input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', '-t', type=str, help='Text to render (use quotes)')
    group.add_argument('--text-file', '-f', type=str, help='Path to a text file to render')

    # Output options
    parser.add_argument('--output', '-o', type=str, default='img/output.png', 
                        help='Output file path (default: img/output.png)')
    parser.add_argument('--format', choices=['svg', 'png'], default='png',
                        help='Output format: svg or png (default: png)')
    
    # Text formatting options
    parser.add_argument('--width', '-w', type=int, default=75,
                        help='Maximum characters per line (default: 75)')
    
    # Style options
    parser.add_argument('--style', type=str, help='Style index or comma-separated list (e.g. 9 or 9,9,12)')
    parser.add_argument('--bias', type=str, default='0.5', help='Bias (float) or comma separated list (default: 0.5)')
    parser.add_argument('--color', type=str, default='black', help='Stroke color or comma-separated list (default: black)')
    parser.add_argument('--stroke_width', type=str, default='2', help='Stroke width or comma-separated list (default: 2)')
    
    # PNG transparency options
    parser.add_argument('--transparency_method', choices=['saturation', 'luminance', 'distance'], 
                        default='distance',
                        help='Method for PNG background removal: saturation (best for colors), luminance (perceptual), distance (mathematical) (default: saturation)')

    args = parser.parse_args()

    # Read text input
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as fh:
            txt = fh.read()
    else:
        txt = args.text

    # Parse arguments
    style_val = _parse_list_arg(args.style)
    bias_val = _parse_list_arg(args.bias)
    color_val = _parse_list_arg(args.color)
    width_val = _parse_list_arg(args.stroke_width)

    # Generate handwriting
    try:
        out = generate_handwriting(
            text=txt,
            output_path=args.output,
            width=args.width,
            bias=bias_val,
            style=style_val,
            color=color_val,
            stroke_width=width_val,
            output_format=args.format,
            transparency_method=args.transparency_method
        )
        print(f"Saved handwriting {args.format.upper()} to: {out}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)