"""
This project is developed by Haofan Wang to support face swap in single frame. Multi-frame will be supported soon!

It is highly built on the top of insightface, sd-webui-roop and CodeFormer.
# """
# import pdb
# import os
# import cv2
# import copy
# import argparse
# import insightface
# import onnxruntime
# import numpy as np
# from PIL import Image
# from typing import List, Union, Dict, Set, Tuple


# def getFaceSwapModel(model_path: str):
#     model = insightface.model_zoo.get_model(model_path, providers=['CUDAExecutionProvider'])
#     return model


# def getFaceAnalyser(model_path: str, providers,
#                     det_size=(320, 320)):
#     face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=['CUDAExecutionProvider'])
#     face_analyser.prepare(ctx_id=0, det_size=det_size)
#     return face_analyser


# def get_one_face(face_analyser,
#                  frame:np.ndarray):
#     face = face_analyser.get(frame)
#     try:
#         return min(face, key=lambda x: x.bbox[0])
#     except ValueError:
#         return None

    
# def get_many_faces(face_analyser,
#                    frame:np.ndarray):
#     """
#     get faces from left to right by order
#     """
#     try:
#         face = face_analyser.get(frame)
#         return sorted(face, key=lambda x: x.bbox[0])
#     except IndexError:
#         return None


# def swap_face(face_swapper,
#               source_faces,
#               target_faces,
#               source_index,
#               target_index,
#               temp_frame):
#     """
#     paste source_face on target image
#     """
#     source_face = source_faces[source_index]
#     target_face = target_faces[target_index]

#     return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
 
    
# def process(source_img: Union[Image.Image, List],
#             target_img: Image.Image,
#             source_indexes: str,
#             target_indexes: str,
#             model: str):
#     # load machine default available providers
#     # providers = onnxruntime.get_available_providers()
#     providers = ['CUDAExecutionProvider'] 
#     print("ONNX Runtime available providers:", providers)
#     # pdb.set_trace()
#     # load face_analyser
#     face_analyser = getFaceAnalyser(model, providers)
    
#     # load face_swapper
#     model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
#     # pdb.set_trace()
#     face_swapper = getFaceSwapModel(model_path)
    
#     # read target image
#     target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
#     # detect faces that will be replaced in the target image
#     target_faces = get_many_faces(face_analyser, target_img)
#     num_target_faces = len(target_faces)
#     num_source_images = len(source_img)

#     if target_faces is not None:
#         temp_frame = copy.deepcopy(target_img)
#         if isinstance(source_img, list) and num_source_images == num_target_faces:
#             print("Replacing faces in target image from the left to the right by order")
#             for i in range(num_target_faces):
#                 source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
#                 source_index = i
#                 target_index = i

#                 if source_faces is None:
#                     raise Exception("No source faces found!")

#                 temp_frame = swap_face(
#                     face_swapper,
#                     source_faces,
#                     target_faces,
#                     source_index,
#                     target_index,
#                     temp_frame
#                 )
#         elif num_source_images == 1:
#             # detect source faces that will be replaced into the target image
#             source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
#             num_source_faces = len(source_faces)
#             print(f"Source faces: {num_source_faces}")
#             print(f"Target faces: {num_target_faces}")

#             if source_faces is None:
#                 raise Exception("No source faces found!")

#             if target_indexes == "-1":
#                 if num_source_faces == 1:
#                     print("Replacing all faces in target image with the same face from the source image")
#                     num_iterations = num_target_faces
#                 elif num_source_faces < num_target_faces:
#                     print("There are less faces in the source image than the target image, replacing as many as we can")
#                     num_iterations = num_source_faces
#                 elif num_target_faces < num_source_faces:
#                     print("There are less faces in the target image than the source image, replacing as many as we can")
#                     num_iterations = num_target_faces
#                 else:
#                     print("Replacing all faces in the target image with the faces from the source image")
#                     num_iterations = num_target_faces

#                 for i in range(num_iterations):
#                     source_index = 0 if num_source_faces == 1 else i
#                     target_index = i

#                     temp_frame = swap_face(
#                         face_swapper,
#                         source_faces,
#                         target_faces,
#                         source_index,
#                         target_index,
#                         temp_frame
#                     )
#             else:
#                 print("Replacing specific face(s) in the target image with specific face(s) from the source image")

#                 if source_indexes == "-1":
#                     source_indexes = ','.join(map(lambda x: str(x), range(num_source_faces)))

#                 if target_indexes == "-1":
#                     target_indexes = ','.join(map(lambda x: str(x), range(num_target_faces)))

#                 source_indexes = source_indexes.split(',')
#                 target_indexes = target_indexes.split(',')
#                 num_source_faces_to_swap = len(source_indexes)
#                 num_target_faces_to_swap = len(target_indexes)

#                 if num_source_faces_to_swap > num_source_faces:
#                     raise Exception("Number of source indexes is greater than the number of faces in the source image")

#                 if num_target_faces_to_swap > num_target_faces:
#                     raise Exception("Number of target indexes is greater than the number of faces in the target image")

#                 if num_source_faces_to_swap > num_target_faces_to_swap:
#                     num_iterations = num_source_faces_to_swap
#                 else:
#                     num_iterations = num_target_faces_to_swap

#                 if num_source_faces_to_swap == num_target_faces_to_swap:
#                     for index in range(num_iterations):
#                         source_index = int(source_indexes[index])
#                         target_index = int(target_indexes[index])

#                         if source_index > num_source_faces-1:
#                             raise ValueError(f"Source index {source_index} is higher than the number of faces in the source image")

#                         if target_index > num_target_faces-1:
#                             raise ValueError(f"Target index {target_index} is higher than the number of faces in the target image")

#                         temp_frame = swap_face(
#                             face_swapper,
#                             source_faces,
#                             target_faces,
#                             source_index,
#                             target_index,
#                             temp_frame
#                         )
#         else:
#             raise Exception("Unsupported face configuration")
#         result = temp_frame
#     else:
#         print("No target faces found!")
    
#     result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
#     return result_image


# def parse_args():
#     parser = argparse.ArgumentParser(description="Face swap.")
#     parser.add_argument("--source_img", type=str, required=True, help="The path of source image, it can be multiple images, dir;dir2;dir3.")
#     parser.add_argument("--target_img", type=str, required=True, help="The path of target image.")
#     parser.add_argument("--output_img", type=str, required=False, default="result.png", help="The path and filename of output image.")
#     parser.add_argument("--source_indexes", type=str, required=False, default="-1", help="Comma separated list of the face indexes to use (left to right) in the source image, starting at 0 (-1 uses all faces in the source image")
#     parser.add_argument("--target_indexes", type=str, required=False, default="-1", help="Comma separated list of the face indexes to swap (left to right) in the target image, starting at 0 (-1 swaps all faces in the target image")
#     parser.add_argument("--face_restore", action="store_true", help="The flag for face restoration.")
#     parser.add_argument("--background_enhance", action="store_true", help="The flag for background enhancement.")
#     parser.add_argument("--face_upsample", action="store_true", help="The flag for face upsample.")
#     parser.add_argument("--upscale", type=int, default=1, help="The upscale value, up to 4.")
#     parser.add_argument("--codeformer_fidelity", type=float, default=0.5, help="The codeformer fidelity.")
#     args = parser.parse_args()
#     return args



import os
import cv2
import copy
import insightface
import numpy as np
from PIL import Image
from typing import List, Union

# -------------------- Global Cache --------------------
_GLOBAL_FACE_ANALYSER = None
_GLOBAL_FACE_SWAPPER = None

def getFaceSwapModel(model_path: str):
    global _GLOBAL_FACE_SWAPPER
    if _GLOBAL_FACE_SWAPPER is None:
        _GLOBAL_FACE_SWAPPER = insightface.model_zoo.get_model(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    return _GLOBAL_FACE_SWAPPER


def getFaceAnalyser(det_size=(320, 320)):
    global _GLOBAL_FACE_ANALYSER
    if _GLOBAL_FACE_ANALYSER is None:
        _GLOBAL_FACE_ANALYSER = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root="./checkpoints",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _GLOBAL_FACE_ANALYSER.prepare(ctx_id=0, det_size=det_size)
    return _GLOBAL_FACE_ANALYSER


def get_one_face(face_analyser, frame: np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyser, frame: np.ndarray):
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


def process(
    source_img: Union[Image.Image, List],
    target_img: Image.Image,
    source_indexes: str,
    target_indexes: str,
    model: str,
    swapper=None,
    analyser=None
):
    """
    Main face swapping logic with cached models.
    """
    # Load cached analyser and swapper if not passed
    face_analyser = analyser if analyser else getFaceAnalyser()
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = swapper if swapper else getFaceSwapModel(model_path)

    # Convert target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    # Detect target faces
    target_faces = get_many_faces(face_analyser, target_img)
    if target_faces is None or len(target_faces) == 0:
        print("No target faces found!")
        return Image.fromarray(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

    temp_frame = copy.deepcopy(target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    # Case 1: multiple sources, same count as target
    if isinstance(source_img, list) and num_source_images == num_target_faces:
        print("Replacing faces in target image from the left to the right by order")
        for i in range(num_target_faces):
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
            if source_faces is None:
                raise Exception("No source faces found!")
            temp_frame = swap_face(face_swapper, source_faces, target_faces, 0, i, temp_frame)

    # Case 2: single source image
    elif num_source_images == 1:
        source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
        if source_faces is None:
            raise Exception("No source faces found!")

        if target_indexes == "-1":
            for i in range(num_target_faces):
                temp_frame = swap_face(face_swapper, source_faces, target_faces, 0, i, temp_frame)
        else:
            source_indexes = source_indexes.split(",")
            target_indexes = target_indexes.split(",")
            for s_idx, t_idx in zip(source_indexes, target_indexes):
                temp_frame = swap_face(face_swapper, source_faces, target_faces, int(s_idx), int(t_idx), temp_frame)
    else:
        raise Exception("Unsupported face configuration")

    result_image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
    return result_image




if __name__ == "__main__":
    
    args = parse_args()
    
    source_img_paths = args.source_img.split(';')
    print("Source image paths:", source_img_paths)
    target_img_path = args.target_img
    
    source_img = [Image.open(img_path) for img_path in source_img_paths]
    target_img = Image.open(target_img_path)

    # download from https://huggingface.co/deepinsight/inswapper/tree/main
    model = "./checkpoints/inswapper_128.onnx"
    result_image = process(source_img, target_img, args.source_indexes, args.target_indexes, model)
    
    if args.face_restore:
        from restoration import *
        
        # make sure the ckpts downloaded successfully
        check_ckpts()
        
        # https://huggingface.co/spaces/sczhou/CodeFormer
        upsampler = set_realesrgan()
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                         codebook_size=1024,
                                                         n_head=8,
                                                         n_layers=9,
                                                         connect_list=["32", "64", "128", "256"],
                                                        ).to(device)
        ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path)["params_ema"]
        codeformer_net.load_state_dict(checkpoint)
        codeformer_net.eval()
        
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_image = face_restoration(result_image, 
                                        args.background_enhance, 
                                        args.face_upsample, 
                                        args.upscale, 
                                        args.codeformer_fidelity,
                                        upsampler,
                                        codeformer_net,
                                        device)
        result_image = Image.fromarray(result_image)
    
    # save result
    result_image.save(args.output_img)
    print(f'Result saved successfully: {args.output_img}')
















import os
import cv2
import copy
import insightface
import numpy as np
from PIL import Image
from typing import List, Union

# -------------------- Global Cache --------------------
_GLOBAL_FACE_ANALYSER = None
_GLOBAL_FACE_SWAPPER = None

def getFaceSwapModel(model_path: str):
    global _GLOBAL_FACE_SWAPPER
    if _GLOBAL_FACE_SWAPPER is None:
        _GLOBAL_FACE_SWAPPER = insightface.model_zoo.get_model(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    return _GLOBAL_FACE_SWAPPER


def getFaceAnalyser(det_size=(320, 320)):
    global _GLOBAL_FACE_ANALYSER
    if _GLOBAL_FACE_ANALYSER is None:
        _GLOBAL_FACE_ANALYSER = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root="./checkpoints",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _GLOBAL_FACE_ANALYSER.prepare(ctx_id=0, det_size=det_size)
    return _GLOBAL_FACE_ANALYSER


def get_one_face(face_analyser, frame: np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyser, frame: np.ndarray):
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


def process(
    source_img: Union[Image.Image, List],
    target_img: Image.Image,
    source_indexes: str,
    target_indexes: str,
    model: str,
    swapper=None,
    analyser=None
):
    """
    Main face swapping logic with cached models.
    """
    # Load cached analyser and swapper if not passed
    face_analyser = analyser if analyser else getFaceAnalyser()
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = swapper if swapper else getFaceSwapModel(model_path)

    # Convert target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    # Detect target faces
    target_faces = get_many_faces(face_analyser, target_img)
    if target_faces is None or len(target_faces) == 0:
        print("No target faces found!")
        return Image.fromarray(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

    temp_frame = copy.deepcopy(target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    # Case 1: multiple sources, same count as target
    if isinstance(source_img, list) and num_source_images == num_target_faces:
        print("Replacing faces in target image from the left to the right by order")
        for i in range(num_target_faces):
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
            if source_faces is None:
                raise Exception("No source faces found!")
            temp_frame = swap_face(face_swapper, source_faces, target_faces, 0, i, temp_frame)

    # Case 2: single source image
    elif num_source_images == 1:
        source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
        if source_faces is None:
            raise Exception("No source faces found!")

        if target_indexes == "-1":
            for i in range(num_target_faces):
                temp_frame = swap_face(face_swapper, source_faces, target_faces, 0, i, temp_frame)
        else:
            source_indexes = source_indexes.split(",")
            target_indexes = target_indexes.split(",")
            for s_idx, t_idx in zip(source_indexes, target_indexes):
                temp_frame = swap_face(face_swapper, source_faces, target_faces, int(s_idx), int(t_idx), temp_frame)
    else:
        raise Exception("Unsupported face configuration")

    result_image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
    return result_image



# this is swaper.py







import cv2
import numpy as np
from PIL import Image
from swapper import (
    process,
    getFaceAnalyser,
    getFaceSwapModel,
    get_many_faces,
)
from gfpgan import GFPGANer

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    # Input/output
    "source_image": "/home/ubuntu/daksh/inswapper/inswapper/data/ekta_source_image.jpeg",
    "target_video": "/home/ubuntu/daksh/inswapper/inswapper/data/WhatsApp Video 2025-09-02 at 13.45.41 1 (online-video-cutter.com).mp4",
    "output_video": "/home/ubuntu/daksh/inswapper/inswapper/outputs/final_best_to_show.mp4",

    # Models
    "swapper_model": "./checkpoints/inswapper_128_fp16.onnx",
    "gfpgan_model": "./checkpoints/GFPGANv1.3.pth",

    # Face tracking
    "target_face_index": 0,   # Which face to swap in first frame (0 = leftmost)
    "det_size": (320, 320),   # Face detector size (smaller = faster, less accurate)

    # Similarity / adaptation
    "similarity_threshold": 0.20,
    "adapt_embedding": True,
    "adapt_rate": 0.05,
    "adapt_confidence": 0.85,

    # Debugging
    "log_interval": 100       # Log progress every N frames
}


# =========================================================
# HELPERS
# =========================================================
def batch_similarity(target_embedding, faces):
    """Vectorized cosine similarity for all detected faces."""
    if not faces:
        return []
    embs = np.stack([f.normed_embedding for f in faces])
    sims = np.dot(embs, target_embedding)  # embeddings are normalized
    return sims


def enhance_with_gfpgan(restorer, frame_bgr):
    """Enhance swapped face using GFPGAN, fallback to original if fails."""
    try:
        _, _, enhanced_bgr = restorer.enhance(
            frame_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        return enhanced_bgr
    except Exception as e:
        print(f"[WARNING] GFPGAN failed: {e}")
        return frame_bgr


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_face_swap(config):
    print("[INFO] Initializing models...")
    analyser = getFaceAnalyser(det_size=config["det_size"])
    swapper_model = getFaceSwapModel(config["swapper_model"])

    restorer = GFPGANer(
        model_path=config["gfpgan_model"],
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None
    )

    # Load source face
    source_img_list = [Image.open(config["source_image"]).convert("RGB")]

    # Open video
    cap = cv2.VideoCapture(config["target_video"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {config['target_video']}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(config["output_video"], fourcc, fps, (width, height))

    print(f"[INFO] Video opened: {frame_count} frames, {width}x{height}@{fps:.2f}fps")

    # ---- Step 1: Pick target face in first frame
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame from video.")

    first_faces = get_many_faces(analyser, first_frame)
    if not first_faces:
        raise RuntimeError("No faces detected in first frame.")

    if config["target_face_index"] >= len(first_faces):
        raise IndexError(
            f"Requested face index {config['target_face_index']} "
            f"but only {len(first_faces)} face(s) found."
        )

    target_face = first_faces[config["target_face_index"]]
    target_embedding = target_face.normed_embedding.copy()

    print(f"[INFO] Selected target face {config['target_face_index']} "
          f"(embedding shape {target_embedding.shape})")

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---- Step 2: Process frames
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        faces = get_many_faces(analyser, frame_bgr)
        if not faces:
            out.write(frame_bgr)
            continue

        # Similarity (vectorized)
        sims = batch_similarity(target_embedding, faces)
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if frame_idx % config["log_interval"] == 0 or best_sim < config["similarity_threshold"]:
            print(f"[INFO] frame {frame_idx}/{frame_count} - best_idx={best_idx}, sim={best_sim:.3f}")

        if best_sim < config["similarity_threshold"]:
            out.write(frame_bgr)
            continue

        # Swap face
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        try:
            swapped_pil = process(
                source_img_list,
                pil_frame,
                source_indexes="0",
                target_indexes=str(best_idx),
                model=config["swapper_model"],
                swapper=swapper_model,
                analyser=analyser
            )
            swapped_bgr = cv2.cvtColor(np.array(swapped_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[WARNING] Swap failed on frame {frame_idx}: {e}")
            out.write(frame_bgr)
            continue

        # Enhance
        enhanced_bgr = enhance_with_gfpgan(restorer, swapped_bgr)
        out.write(enhanced_bgr)

        # Adapt embedding
        if config["adapt_embedding"] and best_sim >= config["adapt_confidence"]:
            new_emb = faces[best_idx].normed_embedding
            target_embedding = (1.0 - config["adapt_rate"]) * target_embedding + config["adapt_rate"] * new_emb
            target_embedding /= np.linalg.norm(target_embedding)

    # ---- Step 3: Cleanup
    cap.release()
    out.release()
    print(f"[INFO] Done. Saved to: {config['output_video']}")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    run_face_swap(CONFIG)
