import numpy as np
import soundfile as sf
from scipy.signal import resample
import pygame

def mark_word_boundaries(audio_path, speed=0.7, window=0.8, step=0.02):
    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    duration = len(y) / sr
    pos = 0.0
    marks = []

    pygame.init()
    pygame.mixer.quit()
    pygame.mixer.init(frequency=sr, size=-16, channels=2)
    screen = pygame.display.set_mode((800, 140))
    pygame.display.set_caption("Audio boundary marker")

    font = pygame.font.SysFont(None, 28)
    small = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    def make_segment(center):
        start = max(0.0, center - window / 2)
        end = min(duration, center + window / 2)
        seg = y[int(start * sr):int(end * sr)]
        if len(seg) == 0:
            return None

        n_new = max(1, int(len(seg) / speed))
        seg_slow = resample(seg, n_new)
        seg_slow = np.clip(seg_slow, -1.0, 1.0)

        pcm = (seg_slow * 32767).astype(np.int16)

        # make stereo: shape (N, 2)
        pcm = np.column_stack([pcm, pcm])

        snd = pygame.sndarray.make_sound(pcm)
        return snd, start, end

    def play_current():
        made = make_segment(pos)
        if made is None:
            return
        snd, start, end = made
        pygame.mixer.stop()
        snd.play()
        print(f"play: cursor={pos:.3f}s window=[{start:.3f}, {end:.3f}]")

    running = True
    while running:
        screen.fill((20, 20, 20))

        txt1 = font.render(f"Cursor: {pos:.3f}s / {duration:.3f}s", True, (230, 230, 230))
        txt2 = small.render(
            "Left/Right: move   Space: play   Enter: mark   Backspace: undo   Q: quit",
            True,
            (180, 180, 180),
        )
        txt3 = small.render(f"Marks: {[round(x, 3) for x in marks]}", True, (120, 220, 120))

        screen.blit(txt1, (20, 20))
        screen.blit(txt2, (20, 60))
        screen.blit(txt3, (20, 95))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    pos = min(duration, pos + step)
                    print(f"cursor: {pos:.3f}s")

                elif event.key == pygame.K_LEFT:
                    pos = max(0.0, pos - step)
                    print(f"cursor: {pos:.3f}s")

                elif event.key == pygame.K_SPACE:
                    play_current()

                elif event.key == pygame.K_RETURN:
                    marks.append(pos)
                    marks.sort()
                    print(f"marked: {pos:.3f}s")

                elif event.key == pygame.K_BACKSPACE:
                    if marks:
                        removed = marks.pop()
                        print(f"removed: {removed:.3f}s")

                elif event.key == pygame.K_q:
                    running = False

        clock.tick(30)

    pygame.mixer.stop()
    pygame.quit()
    return [marks/duration for marks in marks]