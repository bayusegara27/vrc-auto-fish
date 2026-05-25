"""
小游戏会话与流水线处理
======================
抽取 FishingBot 中高度内聚的小游戏运行时逻辑，降低 Bot 主类复杂度。
"""

from __future__ import annotations

import os
import queue
import threading
import time

import cv2

import config
from core.minigame_runtime import DetectionContext, MinigameRuntime, PipelineContext
from utils.i18n import fish_name, t
from utils.logger import log
from utils.overlay_text import overlay_fish_name
from yolo.paths import UNLABELED as YOLO_UNLABELED


class MinigameSession:
    """封装一局小游戏中的上下文、流水线和帧级处理逻辑。"""

    def __init__(self, bot):
        self.bot = bot

    def build_runtime(self, entered_minigame_early: bool) -> MinigameRuntime:
        """初始化一局小游戏运行时状态。"""
        return MinigameRuntime(game_active=entered_minigame_early)

    def build_detection_context(
        self,
        use_yolo: bool,
        skip_success_check: bool,
    ) -> DetectionContext:
        """初始化小游戏检测上下文。"""
        return DetectionContext(
            use_yolo=use_yolo,
            skip_success_check=skip_success_check,
            bar_x_half=config.REGION_X,
            fish_x_half=max(config.REGION_X * 2, 80),
        )

    def initialize_context(self, ctx: DetectionContext):
        """初始化搜索区域、截图信息与首帧调试输出。"""
        bot = self.bot
        bot.pd.reset()
        bot._bar_smooth_cy = None
        bot._bar_locked_cx = None
        bot._progress_debug_saved = False

        screen_orig = bot._grab()
        bot.screen.save_debug(screen_orig, "minigame_start")
        h_orig, w_orig = screen_orig.shape[:2]
        log.info_t("bot.log.captureSize", width=w_orig, height=h_orig)
        bot._show_debug_overlay(screen_orig, status_text="Initializing minigame...")

        if bot._need_rotation:
            log.info_t(
                "bot.log.rotationComp",
                angle=bot._track_angle,
                rotate=-bot._track_angle,
            )
            screen = bot._rotate_for_detection(screen_orig)
        else:
            screen = screen_orig

        ctx.height, ctx.width = screen.shape[:2]
        if ctx.use_yolo:
            ctx.search_region = None
            ctx.bar_search_region = None
            ctx.regions_locked = True
            if config.DETECT_ROI:
                log.info_t(
                    "bot.log.useManualRoi",
                    x=config.DETECT_ROI[0],
                    y=config.DETECT_ROI[1],
                    w=config.DETECT_ROI[2],
                    h=config.DETECT_ROI[3],
                )
            elif bot._auto_roi:
                log.info_t(
                    "bot.log.useAutoRoi",
                    x=bot._auto_roi[0],
                    y=bot._auto_roi[1],
                    w=bot._auto_roi[2],
                    h=bot._auto_roi[3],
                )
            else:
                log.info_t("bot.log.useFullScreenDetect")
        else:
            ctx.search_region, track_cx, ctx.bar_search_region = bot._init_search_region(screen)
            ctx.regions_locked = False
            if track_cx is not None:
                bot._bar_locked_cx = track_cx
                log.info_t("bot.log.preLockTrackX", x=track_cx)
            if ctx.search_region:
                srx, sry, srw, srh = ctx.search_region
                log.info_t(
                    "bot.log.initialFishSearch",
                    x1=srx,
                    x2=srx + srw,
                    y1=sry,
                    y2=sry + srh,
                )
            if ctx.bar_search_region:
                bsx, bsy, bsw, bsh = ctx.bar_search_region
                log.info_t(
                    "bot.log.initialBarSearch",
                    x1=bsx,
                    x2=bsx + bsw,
                    y1=bsy,
                    y2=bsy + bsh,
                )

        return screen_orig, screen

    def start_pipeline(self, ctx: DetectionContext) -> PipelineContext:
        """根据当前模式启动同步/异步检测流水线。"""
        bot = self.bot
        sync_pd_mode = (
            getattr(config, "SYNC_PD_MODE", True)
            and not config.IL_RECORD
            and not config.IL_USE_MODEL
        )
        pipe = PipelineContext(sync_pd_mode=sync_pd_mode)
        if pipe.sync_pd_mode:
            log.info_t("bot.log.legacyMode")
            return pipe

        pipe.frame_q = queue.Queue(maxsize=1)
        pipe.result_q = queue.Queue(maxsize=1)
        pipe.stop_evt = threading.Event()
        pipe.shared_params = {
            "search_region": ctx.search_region,
            "bar_search_region": ctx.bar_search_region,
            "locked_fish_key": ctx.locked_fish_key,
            "locked_fish_scales": ctx.locked_fish_scales,
            "locked_bar_scales": ctx.locked_bar_scales,
            "frame": 0,
            "yolo_roi": config.DETECT_ROI or bot._auto_roi,
            "skip_success": ctx.skip_success_check,
        }
        pipe.params_lock = threading.Lock()
        pipe.capture_thread = threading.Thread(
            target=bot._capture_worker_fn,
            args=(pipe.frame_q, pipe.stop_evt),
            daemon=True,
            name="FishCapture",
        )
        pipe.detect_thread = threading.Thread(
            target=bot._detect_worker_fn,
            args=(
                pipe.frame_q,
                pipe.result_q,
                pipe.stop_evt,
                pipe.shared_params,
                pipe.params_lock,
                ctx.use_yolo,
            ),
            daemon=True,
            name="FishDetect",
        )
        pipe.capture_thread.start()
        pipe.detect_thread.start()
        log.info_t("bot.log.pipelineStarted")
        return pipe

    def stop_pipeline(self, pipe: PipelineContext):
        """停止异步检测流水线。"""
        if pipe.sync_pd_mode:
            return
        pipe.stop_evt.set()
        pipe.capture_thread.join(timeout=1.0)
        pipe.detect_thread.join(timeout=1.0)
        log.info_t("bot.log.pipelineStopped")

    def get_next_detection_result(
        self,
        runtime: MinigameRuntime,
        ctx: DetectionContext,
        pipe: PipelineContext,
    ):
        """获取下一帧检测结果，兼容同步与异步模式。"""
        bot = self.bot
        if pipe.sync_pd_mode:
            next_frame = runtime.frame + 1
            screen_raw = bot._grab()
            screen = (
                bot._rotate_for_detection(screen_raw)
                if bot._need_rotation
                else screen_raw
            )
            bot._tick_fps()
            (
                pipe_fish,
                pipe_bar,
                pipe_progress,
                pipe_hook,
                pipe_mk,
                pipe_bs,
                pipe_track,
                ctx.sync_track_cache,
            ) = bot._detect_frame_once(
                screen,
                ctx.use_yolo,
                ctx.search_region,
                ctx.bar_search_region,
                ctx.locked_fish_key,
                ctx.locked_fish_scales,
                ctx.locked_bar_scales,
                next_frame,
                config.DETECT_ROI or bot._auto_roi,
                ctx.skip_success_check,
                track_cache=ctx.sync_track_cache,
            )
            runtime.frame = next_frame
            return (
                screen_raw,
                screen,
                pipe_fish,
                pipe_bar,
                pipe_progress,
                pipe_hook,
                pipe_mk,
                pipe_bs,
                pipe_track,
            )

        try:
            pipe_data = pipe.result_q.get(timeout=0.5)
        except queue.Empty:
            return None
        while True:
            try:
                pipe_data = pipe.result_q.get_nowait()
            except queue.Empty:
                break
        runtime.frame += 1
        bot._tick_fps()
        return pipe_data

    def sync_pipeline_params(
        self,
        runtime: MinigameRuntime,
        ctx: DetectionContext,
        pipe: PipelineContext,
    ):
        """同步检测参数给异步检测线程。"""
        if pipe.sync_pd_mode:
            return
        with pipe.params_lock:
            pipe.shared_params["search_region"] = ctx.search_region
            pipe.shared_params["bar_search_region"] = ctx.bar_search_region
            pipe.shared_params["locked_fish_key"] = ctx.locked_fish_key
            pipe.shared_params["locked_fish_scales"] = ctx.locked_fish_scales
            pipe.shared_params["locked_bar_scales"] = ctx.locked_bar_scales
            pipe.shared_params["frame"] = runtime.frame

    def get_fish_display(self):
        return {
            key: (overlay_fish_name(key), color)
            for key, color in self.bot.FISH_COLORS.items()
        }

    def reset_fish_name_state(self, runtime: MinigameRuntime):
        """清理鱼类别稳定器与白名单确认状态。"""
        runtime.fish_name_pending = ""
        runtime.fish_name_pending_frames = 0
        runtime.blocked_fish_pending = ""
        runtime.blocked_fish_pending_frames = 0

    def stabilize_fish_name(self, detected_name: str, runtime: MinigameRuntime) -> str:
        """
        对 YOLO 鱼类别做短时稳定，避免同一条鱼在相邻帧里频繁切色。
        返回当前应当生效的稳定类别；空字符串表示暂不接受切换。
        """
        bot = self.bot
        detected_name = detected_name or ""
        if not detected_name:
            runtime.fish_name_pending = ""
            runtime.fish_name_pending_frames = 0
            return bot._current_fish_name or ""

        current_name = bot._current_fish_name or ""
        if detected_name == current_name:
            runtime.fish_name_pending = ""
            runtime.fish_name_pending_frames = 0
            return detected_name

        if runtime.fish_name_pending != detected_name:
            runtime.fish_name_pending = detected_name
            runtime.fish_name_pending_frames = 1
        else:
            runtime.fish_name_pending_frames += 1

        stable_frames = max(1, getattr(config, "YOLO_FISH_STABLE_FRAMES", 3))
        if runtime.fish_name_pending_frames < stable_frames:
            return current_name

        prev_name = current_name
        runtime.fish_name_pending = ""
        runtime.fish_name_pending_frames = 0
        if prev_name and prev_name != detected_name:
            display_map = self.get_fish_display()
            prev_cn = display_map.get(prev_name, (prev_name,))[0]
            new_cn = display_map.get(detected_name, (detected_name,))[0]
            log.info_t(
                "bot.log.fishSwitch",
                old=prev_cn,
                new=new_cn,
                frames=stable_frames,
            )
        return detected_name

    def should_skip_fish_by_whitelist(
        self,
        fish_name_key: str,
        runtime: MinigameRuntime,
    ) -> bool:
        """非白名单鱼需要连续命中多帧才真正触发放弃。"""
        if not fish_name_key:
            runtime.blocked_fish_pending = ""
            runtime.blocked_fish_pending_frames = 0
            return False

        if config.FISH_WHITELIST.get(fish_name_key, True):
            runtime.blocked_fish_pending = ""
            runtime.blocked_fish_pending_frames = 0
            return False

        if runtime.blocked_fish_pending != fish_name_key:
            runtime.blocked_fish_pending = fish_name_key
            runtime.blocked_fish_pending_frames = 1
        else:
            runtime.blocked_fish_pending_frames += 1

        confirm_frames = max(1, getattr(config, "YOLO_WHITELIST_CONFIRM_FRAMES", 6))
        if runtime.blocked_fish_pending_frames < confirm_frames:
            return False

        runtime.blocked_fish_pending = ""
        runtime.blocked_fish_pending_frames = 0
        return True

    def postprocess_detection(
        self,
        screen,
        screen_raw,
        fish,
        bar,
        matched_key,
        bar_scale,
        yolo_progress,
        prog_hook,
        runtime: MinigameRuntime,
        ctx: DetectionContext,
    ):
        """处理一帧检测结果的模板锁定、轨道约束与调试显示。"""
        bot = self.bot
        fish_detect_name = ""

        if ctx.use_yolo:
            if fish is not None:
                fish_detect_name = matched_key or ""
                if not fish_detect_name:
                    save_debug = not runtime.fish_id_saved
                    color_key = bot.detector.identify_fish_type(
                        screen,
                        fish,
                        debug_save=save_debug,
                    )
                    if save_debug:
                        runtime.fish_id_saved = True
                    matched_key = color_key
                    fish_detect_name = color_key
                fish_detect_name = self.stabilize_fish_name(fish_detect_name, runtime)

            if config.YOLO_COLLECT and runtime.frame % 10 == 0:
                os.makedirs(YOLO_UNLABELED, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                ms = int((time.time() % 1) * 1000)
                cv2.imwrite(os.path.join(YOLO_UNLABELED, f"{ts}_{ms:03d}.png"), screen)
        else:
            if ctx.locked_fish_key:
                if fish is not None:
                    fish_detect_name = ctx.locked_fish_key
                if fish is None and runtime.fish_lost > 20 and runtime.fish_lost % 20 == 0:
                    ctx.locked_fish_key = None
                    ctx.locked_fish_scales = None
                    log.info_t("bot.log.unlockFishTemplate")
            elif fish is not None:
                fish_detect_name = matched_key or "?"
                if matched_key and matched_key != "fish_white":
                    ctx.locked_fish_key = matched_key
                    scale = bot.detector._last_best_scale
                    ctx.locked_fish_scales = [
                        round(scale * 0.85, 2),
                        scale,
                        round(scale * 1.15, 2),
                    ]
                    log.info_t(
                        "bot.log.lockFishTemplate",
                        name=ctx.locked_fish_key,
                        scales=[f"{x:.2f}" for x in ctx.locked_fish_scales],
                    )

        if fish is not None:
            bot._current_fish_name = fish_detect_name
        if not runtime.skip_fish and fish_detect_name:
            wl_key = fish_detect_name
            if self.should_skip_fish_by_whitelist(wl_key, runtime):
                fname_cn = self.get_fish_display().get(wl_key, (wl_key,))[0]
                confirm_frames = max(
                    1,
                    getattr(config, "YOLO_WHITELIST_CONFIRM_FRAMES", 6),
                )
                log.info_t(
                    "bot.log.whitelistSkip",
                    fish=fname_cn,
                    frames=confirm_frames,
                )
                runtime.skip_fish = True

        if not ctx.use_yolo and bar is not None and not ctx.locked_bar_scales:
            ctx.locked_bar_scales = [
                round(max(0.2, bar_scale * 0.85), 2),
                bar_scale,
                round(bar_scale * 1.15, 2),
            ]
            log.info_t(
                "bot.log.lockBarTemplate",
                scales=[f"{x:.2f}" for x in ctx.locked_bar_scales],
            )

        if bar is not None:
            raw_bcx = bar[0] + bar[2] // 2
            if bot._bar_locked_cx is None:
                bot._bar_locked_cx = raw_bcx
                log.info_t("bot.log.lockTrackXFromBar", x=raw_bcx)
            elif abs(raw_bcx - bot._bar_locked_cx) > ctx.bar_x_half:
                bar = None

            if bar is not None:
                raw_bar_cy = bar[1] + bar[3] // 2
                if bot._bar_smooth_cy is None:
                    bot._bar_smooth_cy = float(raw_bar_cy)
                else:
                    max_jump = max(12.0, bar[3] * 0.60)
                    delta = raw_bar_cy - bot._bar_smooth_cy
                    if delta > max_jump:
                        raw_bar_cy = int(bot._bar_smooth_cy + max_jump)
                    elif delta < -max_jump:
                        raw_bar_cy = int(bot._bar_smooth_cy - max_jump)
                    bot._bar_smooth_cy = 0.45 * raw_bar_cy + 0.55 * bot._bar_smooth_cy
                smooth_bar_cy = int(round(bot._bar_smooth_cy))
                bar = (
                    bot._bar_locked_cx - bar[2] // 2,
                    smooth_bar_cy - bar[3] // 2,
                    bar[2],
                    bar[3],
                    bar[4],
                )
            else:
                bot._bar_smooth_cy = None

        if bar is not None and not ctx.regions_locked:
            bar_cy = bar[1] + bar[3] // 2
            tcx = bot._bar_locked_cx or (bar[0] + bar[2] // 2)
            y_top = max(0, bar_cy - config.REGION_UP)
            y_bot = min(ctx.height, bar_cy + config.REGION_DOWN)
            roi = config.DETECT_ROI
            if roi:
                y_top = max(y_top, roi[1])
                y_bot = min(y_bot, roi[1] + roi[3])
            rh = y_bot - y_top

            fish_half = max(config.REGION_X * 2, 80)
            fsx = max(0, tcx - fish_half)
            fsw = min(fish_half * 2, ctx.width - fsx)
            if roi:
                fsx = max(fsx, roi[0])
                fsw = min(fsw, roi[0] + roi[2] - fsx)
            ctx.search_region = (fsx, y_top, fsw, rh)

            bar_half = config.REGION_X
            bsx = max(0, tcx - bar_half)
            bsw = min(bar_half * 2, ctx.width - bsx)
            if roi:
                bsx = max(bsx, roi[0])
                bsw = min(bsw, roi[0] + roi[2] - bsx)
            ctx.bar_search_region = (bsx, y_top, bsw, rh)
            ctx.regions_locked = True
            log.info(
                f"  ★ 搜索区域锁定(白条Y={bar_cy}): "
                f"Y={y_top}~{y_bot} "
                f"鱼X=±{fish_half} 条X=±{bar_half}"
                f"{' (ROI裁剪)' if roi else ''}"
            )

        if fish is not None:
            raw_fcx = fish[0] + fish[2] // 2
            if bot._bar_locked_cx is not None and abs(raw_fcx - bot._bar_locked_cx) > ctx.fish_x_half:
                fish = None
                bot._current_fish_name = ""
                self.reset_fish_name_state(runtime)
            if fish is not None and bot._bar_locked_cx is not None:
                fish = (
                    bot._bar_locked_cx - fish[2] // 2,
                    fish[1],
                    fish[2],
                    fish[3],
                    fish[4],
                )

        if fish is not None and bar is not None:
            fish_cy_check = fish[1] + fish[3] // 2
            bar_cy_check = bar[1] + bar[3] // 2
            dist_y = abs(fish_cy_check - bar_cy_check)
            if dist_y > config.MAX_FISH_BAR_DIST:
                if runtime.frame % 30 == 1:
                    log.warning(
                        f"[⚠ 误检] 鱼Y={fish_cy_check} 条Y={bar_cy_check} "
                        f"距离={dist_y}px > {config.MAX_FISH_BAR_DIST}px"
                    )
                fish = None
                bar = None

        display_sr = ctx.search_region or bot._auto_roi
        if not bot._need_rotation:
            bot._show_debug_overlay(
                screen_raw,
                fish,
                bar,
                display_sr,
                bar_search_region=ctx.bar_search_region,
                progress=None if ctx.skip_success_check else yolo_progress,
                prog_hook=prog_hook,
                status_text=f"Minigame F{runtime.frame:04d}",
            )
        else:
            bot._show_debug_overlay(
                screen_raw,
                search_region=display_sr,
                bar_search_region=ctx.bar_search_region,
                progress=None if ctx.skip_success_check else yolo_progress,
                prog_hook=prog_hook,
                status_text=(
                    f"Minigame F{runtime.frame:04d} "
                    f"(rotation {bot._track_angle:.0f} deg)"
                ),
            )

        return fish, bar, yolo_progress

    def compute_progress(
        self,
        screen,
        screen_raw,
        fish,
        bar,
        yolo_progress,
        prog_hook,
        runtime: MinigameRuntime,
        ctx: DetectionContext,
    ) -> float:
        """统计当前进度条绿色占比。"""
        bot = self.bot
        green = 0.0
        if ctx.skip_success_check or runtime.frame <= runtime.progress_skip_frames:
            return green

        if ctx.use_yolo and yolo_progress is not None:
            px, py, pw, ph = yolo_progress[:4]
            green, hook_box, hook_source = bot.detector.estimate_progress_in_box(
                screen,
                yolo_progress,
            )
            if hook_box is not None:
                prog_hook = hook_box
                if runtime.frame % 10 == 0:
                    hx, hy, hw, hh, hconf = hook_box
                    log.info(
                        f"[Hook] F{runtime.frame:04d} progress=({px},{py},{pw},{ph}) "
                        f"{hook_source}=({hx},{hy},{hw},{hh}) conf={hconf:.2f} "
                        f"ratio={green:.0%}"
                    )
            else:
                pad_x = max(1, int(pw * 0.08))
                pad_y = max(1, int(ph * 0.05))
                sx = px + pad_x
                sy = py + pad_y
                sw = max(1, pw - pad_x * 2)
                sh = max(1, ph - pad_y * 2)
                green = bot.detector.detect_green_ratio(screen, (sx, sy, sw, sh))
                if not bot._progress_debug_saved and green > 0:
                    bot._progress_debug_saved = True
                    pad = 20
                    dx = max(0, px - pad)
                    dw = min(pw + pad * 2, ctx.width - dx)
                    dbg = screen[py:py + ph, dx:dx + dw].copy()
                    cv2.rectangle(
                        dbg,
                        (sx - dx, sy - py),
                        (sx - dx + sw, sy - py + sh),
                        (0, 255, 0),
                        1,
                    )
                    info = f"green={green:.0%} roi={sw}x{sh}"
                    cv2.putText(
                        dbg,
                        info,
                        (2, 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )
                    debug_dir = os.path.join(config.BASE_DIR, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(debug_dir, "progress_strip.png"), dbg)

            display_sr = ctx.search_region or bot._auto_roi
            bot._show_debug_overlay(
                screen_raw,
                fish,
                bar,
                display_sr,
                bar_search_region=ctx.bar_search_region,
                progress=yolo_progress,
                prog_hook=prog_hook,
                status_text=f"Minigame F{runtime.frame:04d}",
            )
        else:
            progress_sr = ctx.search_region
            if bar is not None:
                bcx = bar[0] + bar[2] // 2
                bcy = bar[1] + bar[3] // 2
                pr_half_x = max(config.REGION_X * 2, 80)
                pr_x = max(0, bcx - pr_half_x)
                pr_y = max(0, bcy - config.REGION_UP)
                pr_w = min(pr_half_x * 2, ctx.width - pr_x)
                pr_h = min(config.REGION_UP + config.REGION_DOWN, ctx.height - pr_y)
                progress_sr = (pr_x, pr_y, pr_w, pr_h)
                runtime.last_progress_sr = progress_sr
            elif runtime.last_progress_sr is not None:
                progress_sr = runtime.last_progress_sr
            green = bot._check_progress(screen, fish, progress_sr)

        if green > 0 and runtime.prev_green > 0.01 and (green - runtime.prev_green) > 0.30:
            capped_green = min(green, runtime.prev_green + 0.12)
            log.debug(
                f"  进度跳变过大 {runtime.prev_green:.0%}→{green:.0%}，"
                f"限幅到 {capped_green:.0%}"
            )
            green = capped_green

        if green > 0:
            runtime.prev_green = green
        if green > runtime.last_green:
            runtime.last_green = green
        return green

    def maybe_activate(
        self,
        fish,
        bar,
        yolo_progress,
        runtime: MinigameRuntime,
        ctx: DetectionContext,
    ):
        """检查是否正式进入小游戏控制阶段。"""
        bot = self.bot
        if runtime.game_active:
            return "ok"

        det_count = ((fish is not None) + (bar is not None) + (yolo_progress is not None))
        if det_count >= 2:
            runtime.game_active = True
            runtime.had_good_detection = True
            det_names = []
            if fish is not None:
                det_names.append("鱼")
            if bar is not None:
                det_names.append("条")
            if yolo_progress is not None:
                det_names.append("进度条")
            log.info_t("bot.log.minigameConfirmed", names="+".join(det_names))
            if not config.IL_RECORD:
                press_t = getattr(config, "INITIAL_PRESS_TIME", 0.2)
                bot.input.mouse_down()
                time.sleep(press_t)
                bot.input.mouse_up()
            return "ok"

        if time.time() - runtime.hook_time > ctx.hook_detect_timeout:
            log.warning_t("bot.log.hookTimeout", seconds=ctx.hook_detect_timeout)
            runtime.hook_timeout_retry = True
            return "break"

        return "continue"

    def run_control(
        self,
        fish,
        bar,
        yolo_progress,
        runtime: MinigameRuntime,
        ctx: DetectionContext,
    ) -> bool:
        """执行当前帧控制逻辑。"""
        bot = self.bot
        backend = getattr(bot, "_active_control_backend", None) or bot._build_control_backend()
        return backend.control(fish, bar, yolo_progress, runtime, ctx)

    def log_frame(
        self,
        fish,
        bar,
        green,
        runtime: MinigameRuntime,
        skip_success_check: bool,
    ):
        """输出小游戏周期日志。"""
        bot = self.bot
        if runtime.frame == 50:
            bot.detector.debug_report = bot.debug_mode
        if runtime.frame % 30 != 0:
            return
        fname = bot._current_fish_name.replace("fish_", "") if bot._current_fish_name else ""
        fi = f"鱼[{fname}]Y={fish[1] + fish[3] // 2}" if fish else "鱼=无"
        bi = f"条Y={bar[1] + bar[3] // 2}" if bar else "条=无"
        vel = f"v={bot.pd.bar_velocity:+.0f}"
        if skip_success_check:
            log.info(f"[F{runtime.frame:04d}] {fi} | {bi} | {vel} | 按住:{runtime.hold_count}")
            return
        log.info(
            f"[F{runtime.frame:04d}] {fi} | {bi} | {vel} | "
            f"按住:{runtime.hold_count} | 进度:{green:.0%}"
        )
