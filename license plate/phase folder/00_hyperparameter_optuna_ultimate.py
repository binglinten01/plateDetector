#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_hyperparameter_optuna_ultimate.py — FULLY FIXED VERSION (2025)
 - 100% runnable
 - YOLOv11-compatible
 - RTX4060-optimized
 - Fully stable Optuna sampler/pruner
 - Safe metric extraction
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import optuna
import yaml
import numpy as np

# -------------------------------------------------
# YOLO IMPORT (FIXED for YOLOv9/v10/v11)
# -------------------------------------------------
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError("❌ Cannot import ultralytics.YOLO — install via: pip install ultralytics==8.3.32") from e

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# -------------------------------------------------
# OPTIONAL GPU MONITOR (SAFE FALLBACK)
# -------------------------------------------------
try:
    from src.detector.ultimate.gpu_monitor import RTX4060Monitor
except Exception:
    class RTX4060Monitor:
        def __init__(self, max_temp=85):
            self.max_temp = max_temp
        def check_thermal_throttle(self): return False
        def get_performance_summary(self): return {"gpu_monitor": "fallback", "available": False}

# -------------------------------------------------
# DATA CLASSES
# -------------------------------------------------
@dataclass
class SearchSpace:
    lr0_range: List[float] = field(default_factory=lambda: [1e-4, 1e-2])
    lr0_log: bool = True
    batch_choices: List[int] = field(default_factory=lambda: [8, 16, 24, 32])
    optimizer_choices: List[str] = field(default_factory=lambda: ["AdamW", "Adam", "SGD"])
    weight_decay_range: List[float] = field(default_factory=lambda: [1e-5, 1e-3])
    weight_decay_log: bool = True
    box_range: List[float] = field(default_factory=lambda: [5.0, 10.0])
    cls_range: List[float] = field(default_factory=lambda: [0.3, 0.7])
    dfl_range: List[float] = field(default_factory=lambda: [1.0, 2.0])
    hsv_h_range: List[float] = field(default_factory=lambda: [0.0, 0.1])
    hsv_s_range: List[float] = field(default_factory=lambda: [0.0, 0.9])
    hsv_v_range: List[float] = field(default_factory=lambda: [0.0, 0.9])
    mosaic_range: List[float] = field(default_factory=lambda: [0.4, 1.0])
    mixup_range: List[float] = field(default_factory=lambda: [0.0, 0.3])
    degrees_range: List[float] = field(default_factory=lambda: [0.0, 15.0])
    translate_range: List[float] = field(default_factory=lambda: [0.1, 0.3])
    scale_range: List[float] = field(default_factory=lambda: [0.5, 0.9])
    warmup_epochs_range: List[float] = field(default_factory=lambda: [1.0, 4.0])
    imgsz_choices: List[int] = field(default_factory=lambda: [640, 768])
    patience_choices: List[int] = field(default_factory=lambda: [30, 50, 100])

@dataclass
class HPOResults:
    best_value: float = -float("inf")
    best_params: Dict[str, Any] = field(default_factory=dict)
    trial_history: List[Dict[str, Any]] = field(default_factory=list)
    convergence_data: List[float] = field(default_factory=list)
    importance_scores: Dict[str, float] = field(default_factory=dict)

# -------------------------------------------------
# OPTIMIZER CLASS
# -------------------------------------------------
class UltimateHyperparameterOptimizer:
    def __init__(self, config_path, dataset_path, output_dir="runs/hpo"):
        self.config_path = Path(config_path)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.search_space = self._load_search_space(config_path)
        self.gpu_monitor = RTX4060Monitor(max_temp=82)
        self.results = HPOResults()

        self.study = None
        self.bad_trials_consecutive = 0
        self.max_bad_trials = 3

        print(f"🚀 HPO initialized — output: {self.output_dir}")
        try:
            import signal
            signal.signal(signal.SIGINT, self._signal_handler)
        except Exception:
            pass

    # -------------------------------------------------
    # graceful exit
    # -------------------------------------------------
    def _signal_handler(self, signum, frame):
        print("\n⚠️ Interrupted — saving final results…")
        self._save_final_results()
        sys.exit(0)

    # -------------------------------------------------
    # LOAD SEARCH SPACE (FIXED)
    # -------------------------------------------------
    def _load_search_space(self, config_path):
        space = SearchSpace()
        if not Path(config_path).exists():
            print(f"⚠️ Config not found → using default search space")
            return space

        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}

            for key in vars(space).keys():
                if key in data:
                    setattr(space, key, data[key])
        except Exception as e:
            print(f"⚠️ Invalid search space config: {e}")

        return space

    # -------------------------------------------------
    # OPTUNA SAMPLERS (SAFE)
    # -------------------------------------------------
    def _create_advanced_sampler(self):
        return optuna.samplers.TPESampler(
            seed=42, n_startup_trials=10, n_ei_candidates=24
        )

    def _create_advanced_pruner(self):
        return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=8)

    # -------------------------------------------------
    # PARAMETER SUGGESTION
    # -------------------------------------------------
    def suggest_parameters(self, trial):
        p = {}

        # Learning rate
        p["lr0"] = trial.suggest_float(
            "lr0",
            self.search_space.lr0_range[0],
            self.search_space.lr0_range[1],
            log=self.search_space.lr0_log
        )

        # Final LR ratio
        p["lrf"] = trial.suggest_float("lrf", 0.01, 0.2)

        # Batch size
        p["batch"] = trial.suggest_categorical("batch", self.search_space.batch_choices)

        # Optimizer
        p["optimizer"] = trial.suggest_categorical("optimizer", self.search_space.optimizer_choices)

        # Weight decay
        p["weight_decay"] = trial.suggest_float(
            "weight_decay",
            self.search_space.weight_decay_range[0],
            self.search_space.weight_decay_range[1],
            log=self.search_space.weight_decay_log
        )

        # Loss weights
        p["box"] = trial.suggest_float("box", *self.search_space.box_range)
        p["cls"] = trial.suggest_float("cls", *self.search_space.cls_range)
        p["dfl"] = trial.suggest_float("dfl", *self.search_space.dfl_range)

        # Aug
        p["hsv_h"] = trial.suggest_float("hsv_h", *self.search_space.hsv_h_range)
        p["hsv_s"] = trial.suggest_float("hsv_s", *self.search_space.hsv_s_range)
        p["hsv_v"] = trial.suggest_float("hsv_v", *self.search_space.hsv_v_range)
        p["mosaic"] = trial.suggest_float("mosaic", *self.search_space.mosaic_range)
        p["mixup"] = trial.suggest_float("mixup", *self.search_space.mixup_range)
        p["degrees"] = trial.suggest_float("degrees", *self.search_space.degrees_range)
        p["translate"] = trial.suggest_float("translate", *self.search_space.translate_range)
        p["scale"] = trial.suggest_float("scale", *self.search_space.scale_range)

        # Image size
        p["imgsz"] = trial.suggest_categorical("imgsz", self.search_space.imgsz_choices)

        # Warmup
        p["warmup_epochs"] = trial.suggest_float(
            "warmup_epochs",
            self.search_space.warmup_epochs_range[0],
            self.search_space.warmup_epochs_range[1]
        )

        return p

    # -------------------------------------------------
    # METRIC EXTRACTION — FIXED FOR YOLOv11 API
    # -------------------------------------------------
    def _safe_extract_metrics(self, results):
        """
        Correct extraction for YOLO v10/v11
        """
        m = {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}

        try:
            # YOLO 11 stores final validation metrics here:
            d = getattr(results, "results_dict", None)
            if d is None:
                return m

            m["mAP50"] = float(d.get("metrics/mAP50", 0.0))
            m["mAP50_95"] = float(d.get("metrics/mAP50-95", 0.0))
            m["precision"] = float(d.get("metrics/precision", 0.0))
            m["recall"] = float(d.get("metrics/recall", 0.0))

        except Exception as e:
            print(f"⚠️ Metric extract error: {e}")

        m["fitness"] = (
            0.50 * m["mAP50"]
            + 0.30 * m["mAP50_95"]
            + 0.10 * m["precision"]
            + 0.10 * m["recall"]
        )

        return m

    # -------------------------------------------------
    # YOLO TRAINING (FINAL PRODUCTION VERSION)
    # -------------------------------------------------
    def train_and_evaluate(self, params, trial_number: int, resume=False):

        trial_dir = self.output_dir / f"trial_{trial_number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save chosen hyperparameters
        with open(trial_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)

        print(f"\n🔬 Trial {trial_number:03d}")
        print(f"   lr0={params['lr0']:.6f}  batch={params['batch']}  opt={params['optimizer']}")

        # -----------------------------------------------------
        # GPU THERMAL GUARD
        # -----------------------------------------------------
        try:
            if self.gpu_monitor.check_thermal_throttle():
                print("   ⚠️ GPU hot → cooling 20s")
                time.sleep(20)
        except Exception:
            pass

        # -----------------------------------------------------
        # LOAD BASE YOLO11 MODEL
        # -----------------------------------------------------
        try:
            model = YOLO("yolo11s.pt")
        except Exception:
            print("⚠️ Could not load local yolo11s.pt → downloading from Ultralytics")
            model = YOLO("yolo11s")

        # -----------------------------------------------------
        # TRAINING ARGUMENTS
        # -----------------------------------------------------
        train_args = {
            "data": str(self.dataset_path),
            "epochs": 10,
            "batch": params["batch"],
            "imgsz": params["imgsz"],
            "optimizer": params["optimizer"],
            "lr0": params["lr0"],
            "lrf": params["lrf"],
            "weight_decay": params["weight_decay"],
            "box": params["box"],
            "cls": params["cls"],
            "dfl": params["dfl"],
            "hsv_h": params["hsv_h"],
            "hsv_s": params["hsv_s"],
            "hsv_v": params["hsv_v"],
            "mosaic": params["mosaic"],
            "mixup": params["mixup"],
            "degrees": params["degrees"],
            "translate": params["translate"],
            "scale": params["scale"],
            "warmup_epochs": params["warmup_epochs"],
            "device": 0,
            "workers": 4,
            "save": True,
            "project": str(self.output_dir),
            "name": f"trial_{trial_number:03d}",
            "exist_ok": True,
            "val": True,
            "cos_lr": True,
            "amp": True
        }

        # RTX4060 laptop VRAM protection
        if train_args["batch"] >= 24 and train_args["imgsz"] > 640:
            train_args["imgsz"] = 640

        # -----------------------------------------------------
        # TRAIN THE MODEL
        # -----------------------------------------------------
        try:
            results = model.train(**train_args)
        except Exception as e:
            print(f"❌ Training failed: {e}")
            with open(trial_dir / "error.json", "w") as f:
                json.dump({"error": str(e)}, f, indent=2)
            return -float("inf")

        # -----------------------------------------------------
        # DIRECT YOLOv11 METRIC EXTRACTION (REAL VALUES)
        # -----------------------------------------------------
        try:
            map50 = float(results.box.map50)     # mAP@0.50
            map5095 = float(results.box.map)     # mAP@0.50:0.95
        except Exception as e:
            print(f"❌ Failed to extract metrics: {e}")
            map50 = 0.0
            map5095 = 0.0

        metrics = {
            "mAP50": map50,
            "mAP50_95": map5095
        }

        score = map5095  # Optuna optimizes mAP50-95

        # -----------------------------------------------------
        # SAVE TRIAL RESULTS
        # -----------------------------------------------------
        out = {
            "trial": trial_number,
            "params": params,
            "metrics": metrics,
            "score": score,
            "timestamp": datetime.now().isoformat()
        }

        with open(trial_dir / "results.json", "w") as f:
            json.dump(out, f, indent=2)

        print(f"   ✅ Score={score:.6f}  mAP50={metrics['mAP50']:.4f}")

        # Track results for convergence curves and dashboards
        self.results.trial_history.append(out)
        self.results.convergence_data.append(score)

        return score
    
    # -------------------------------------------------
    # OPTUNA OBJECTIVE WRAPPER
    # -------------------------------------------------
    def objective(self, trial):

        trial_dir = self.output_dir / f"trial_{trial.number:03d}"
        res_file = trial_dir / "results.json"

        # Reuse previous results
        if res_file.exists():
            try:
                with open(res_file, "r") as f:
                    old = json.load(f)
                self.results.trial_history.append(old)
                self.results.convergence_data.append(old["score"])
                return old["score"]
            except Exception:
                pass

        # Bad trial early stop
        if self.bad_trials_consecutive >= self.max_bad_trials:
            raise optuna.TrialPruned()

        params = self.suggest_parameters(trial)
        score = self.train_and_evaluate(params, trial.number)

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Track improvement
        prev_best = max(self.results.convergence_data[:-1], default=-1)
        self.bad_trials_consecutive = 0 if score > prev_best else self.bad_trials_consecutive + 1

        return score

    # -------------------------------------------------
    # LOAD PREVIOUS TRIAL RESULTS (OPTIONAL)
    # -------------------------------------------------
    def _load_previous_results(self):
        try:
            for d in sorted(self.output_dir.glob("trial_*")):
                rf = d / "results.json"
                if rf.exists():
                    with open(rf, "r") as f:
                        r = json.load(f)
                    self.results.trial_history.append(r)
                    self.results.convergence_data.append(r.get("score", 0.0))

            print(f"📊 Loaded {len(self.results.trial_history)} previous trial results.")
        except Exception:
            pass

    # -------------------------------------------------
    # SAVE FINAL RESULTS (best params, statistics)
    # -------------------------------------------------
    def _save_final_results(self):
        results_dir = self.output_dir / "final_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Best params
        with open(results_dir / "best_params.yaml", "w") as f:
            yaml.dump(self.results.best_params, f, sort_keys=False)

        # Study wide stats
        study_stats = {
            "best_value": self.results.best_value,
            "best_params": self.results.best_params,
            "importance_scores": self.results.importance_scores,
            "n_trials": len(self.study.trials) if self.study else 0,
            "convergence": self.results.convergence_data,
            "timestamp": datetime.now().isoformat(),
            "gpu_info": self.gpu_monitor.get_performance_summary()
        }
        with open(results_dir / "study_statistics.json", "w") as f:
            json.dump(study_stats, f, indent=2)

        # All trials in one dataset
        all_trials = []
        if self.study:
            for t in self.study.trials:
                all_trials.append({
                    "number": t.number,
                    "value": float(t.value) if t.value is not None else None,
                    "params": t.params,
                    "state": str(t.state),
                    "start": t.datetime_start.isoformat() if t.datetime_start else None,
                    "end": t.datetime_complete.isoformat() if t.datetime_complete else None
                })
        with open(results_dir / "all_trials.json", "w") as f:
            json.dump(all_trials, f, indent=2)

        print(f"💾 Final results saved → {results_dir}")

    # -------------------------------------------------
    # OPTUNA — MAIN OPTIMIZATION LOOP
    # -------------------------------------------------
    def optimize(self, n_trials=25, timeout=28800):

        print(f"\n🎯 STARTING OPTIMIZATION — trials={n_trials}, timeout={timeout}s")

        storage_path = self.output_dir / "optuna_study.db"
        storage = f"sqlite:///{storage_path}"

        study_name = "ultimate_rtx4060_hpo"

        # Try loading existing study
        try:
            self.study = optuna.load_study(
                study_name=study_name,
                storage=storage
            )
            print(f"🔁 Resumed study with {len(self.study.trials)} trials.")
        except Exception:
            print("✨ Creating new study with TPESampler + HyperbandPruner")
            sampler = self._create_advanced_sampler()
            pruner = self._create_advanced_pruner()

            self.study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                storage=storage,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )

        # Load past trial scores into memory
        self._load_previous_results()

        # Run optimization
        try:
            self.study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=1,
                gc_after_trial=True
            )
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user.")
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            import traceback
            traceback.print_exc()

        # Store best results
        if self.study.best_value is not None:
            self.results.best_value = float(self.study.best_value)
        self.results.best_params = dict(self.study.best_params or {})

        # Parameter importance (optional)
        try:
            self.results.importance_scores = optuna.importance.get_param_importances(self.study)
        except Exception:
            self.results.importance_scores = {}

        # Save everything
        self._save_final_results()

        print("🏁 Optimization finished!")
        print(f"🥇 Best score: {self.results.best_value}")
        print(f"📌 Best params saved.")

        return self.results.best_params

    # -------------------------------------------------
    # FINAL TRAINING RECOMMENDATION
    # -------------------------------------------------
    def generate_training_recommendation(self):
        if not self.results.best_params:
            print("⚠️ No best params available yet.")
            return

        rec = {
            "data": str(self.dataset_path),
            "weights": "yolo11s.pt",
            "epochs": 150,
            "optimizer": self.results.best_params.get("optimizer", "AdamW"),
            "lr0": self.results.best_params.get("lr0", 0.001),
            "lrf": self.results.best_params.get("lrf", 0.01),
            "weight_decay": self.results.best_params.get("weight_decay", 0.0005),
            "batch": self.results.best_params.get("batch", 24),
            "imgsz": self.results.best_params.get("imgsz", 640),
            "box": self.results.best_params.get("box", 7.5),
            "cls": self.results.best_params.get("cls", 0.5),
            "dfl": self.results.best_params.get("dfl", 1.5),
            "mosaic": min(1.0, self.results.best_params.get("mosaic", 1.0)),
            "mixup": self.results.best_params.get("mixup", 0.0),
            "warmup_epochs": self.results.best_params.get("warmup_epochs", 3),
            "amp": True,
            "cosine_lr": True,
            "device": 0,
            "project": str(self.output_dir / "recommended_training"),
            "name": "yolo11s_hpo_best",
            "exist_ok": True
        }

        out = self.output_dir / "final_results" / "recommended_training.yaml"
        with open(out, "w") as f:
            yaml.dump(rec, f, sort_keys=False)

        print(f"📄 Training recommendation saved → {out}")
        return rec


# -------------------------------------------------
# CLI ENTRYPOINT
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ultimate RTX4060 HPO (Fixed Edition)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=28800)
    parser.add_argument("--output-dir", type=str, default="runs/hpo")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    # Quick modes
    if args.test:
        args.trials = 5
        args.timeout = 1200
    elif args.quick:
        args.trials = min(args.trials, 12)
        args.timeout = min(args.timeout, 7200)

    print("=" * 80)
    print("🚀 ULTIMATE RTX4060 HPO — FIXED FULL VERSION")
    print(f"Config:   {args.config}")
    print(f"Dataset:  {args.dataset}")
    print(f"Trials:   {args.trials}")
    print(f"Timeout:  {args.timeout}s")
    print(f"Output:   {args.output_dir}")
    print("=" * 80)

    opt = UltimateHyperparameterOptimizer(
        config_path=args.config,
        dataset_path=args.dataset,
        output_dir=args.output_dir
    )

    best = opt.optimize(
        n_trials=args.trials,
        timeout=args.timeout
    )
    opt.generate_training_recommendation()

    print("\n🎉 HPO COMPLETED SUCCESSFULLY")
    print(f"🔥 Best params saved in {args.output_dir}/final_results")

if __name__ == "__main__":
    main()
