"""폴더 안의 이미지들을 PaddleOCR로 읽어 추출 텍스트를 보여준다.

프로젝트 루트(capturemate-ai)에서 실행:
    python scripts/ocr_run.py                 # 기본 폴더 ./ocr_samples
    python scripts/ocr_run.py --images ./내폴더

각 이미지의 추출 텍스트를 콘솔에 출력하고, ocr_output.txt 로도 저장한다.
ML Kit 결과와의 비교는 직접 눈으로 하면 된다.
"""
import argparse
import sys
import time
from pathlib import Path

# 스크립트 위치와 무관하게 `app` 패키지를 import 할 수 있도록 루트를 경로에 추가.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="PaddleOCR로 이미지 폴더 읽기")
    parser.add_argument("--images", default="./ocr_samples", help="스크린샷 폴더 경로")
    args = parser.parse_args()

    folder = Path(args.images)
    if not folder.is_dir():
        print(f"폴더가 없습니다: {folder.resolve()}")
        return

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"이미지가 없습니다: {folder.resolve()}")
        return

    from app.ocr.paddle_engine import PaddleOcrEngine

    print("PaddleOCR 모델 로딩 중... (최초 실행 시 다운로드로 느릴 수 있음)\n")
    engine = PaddleOcrEngine()

    saved_lines: list[str] = []
    for path in images:
        t0 = time.perf_counter()
        text = engine.extract_text(path.read_bytes())
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        header = f"===== {path.name}  ({elapsed_ms} ms) ====="
        print(header)
        print(text if text.strip() else "(추출된 텍스트 없음)")
        print()

        saved_lines.append(header)
        saved_lines.append(text)
        saved_lines.append("")

    out = Path("ocr_output.txt")
    out.write_text("\n".join(saved_lines), encoding="utf-8")
    print(f"전체 결과 저장: {out.resolve()}")


if __name__ == "__main__":
    main()
