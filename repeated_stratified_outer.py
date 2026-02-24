from sklearn.model_selection import RepeatedStratifiedKFold
from joblib import Parallel, delayed
import pandas as pd
import train_classifier as tc

# ============================================================
# STRUCTURAL CHANGE:
# Instead of single train/test split,
# we now use repeated outer CV to reduce variance.
# ============================================================

# Feature lists
confidence_scores = ['global_plddt', 'mean_plddt_A', 'mean_plddt_B', 'ptm',
       'iptm', 'combined_score', 'ipSAE_max', 'pDockQ_max', 'pDockQ2_max',
       'LIS_max', 'ipsae_A_B', 'pdockq2_A_B', 'lis_A_B', 'ipsae_B_A',
       'pdockq2_B_A', 'lis_B_A']
pyrosetta_scores = ['interface_dG', 'interface_SASA',
       'interface_dG_SASA_ratio', 'shape_complementarity',
       'num_interface_residues', 'buried_unsatisfied_hbonds', 'packstat',
       'binder_dG', 'binder_SASA']
electrostatics = ['idp_core_charged',
       'receptor_core_charged', 'idp_interface_charged',
       'receptor_interface_charged', 'idp_rim_charged', 'receptor_rim_charged',
       'idp_surface_charged', 'receptor_surface_charged',
       'idp_core_hydropathy', 'receptor_core_hydropathy',
       'idp_interface_hydropathy', 'receptor_interface_hydropathy',
       'idp_rim_hydropathy', 'receptor_rim_hydropathy',
       'idp_surface_hydropathy', 'receptor_surface_hydropathy',
       'idp_core_hydrophobic', 'receptor_core_hydrophobic',
       'idp_interface_hydrophobic', 'receptor_interface_hydrophobic',
       'idp_rim_hydrophobic', 'receptor_rim_hydrophobic',
       'idp_surface_hydrophobic', 'receptor_surface_hydrophobic',
       'idp_core_polar', 'receptor_core_polar', 'idp_interface_polar',
       'receptor_interface_polar', 'idp_rim_polar', 'receptor_rim_polar',
       'idp_surface_polar', 'receptor_surface_polar', 'idp_core_small',
       'receptor_core_small', 'idp_interface_small',
       'receptor_interface_small', 'idp_rim_small', 'receptor_rim_small',
       'idp_surface_small', 'receptor_surface_small']
all_features = confidence_scores + pyrosetta_scores + electrostatics

# Define sections and criteria
sections = ["FD", "AFminD (30)", "AFminD (50)", "AFminD (100)"]
criteria = {
    "Min. 5 contacts": 5,
    "40% interacting residues": 0.4,
    "50% interacting residues": 0.5,
    "80% interacting residues": 0.8,
    "40% overlap interface": 0.4,
    "50% overlap interface": 0.5,
    "80% overlap interface": 0.8
}

if __name__ == '__main__':
    # Load dataset once
    print("Loading dataset...")
    df = pd.read_csv("./data/uniprot_idp_domain_receptor_full_dataset_v2.tsv", sep='\t')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

    results = []
    feature_importance_records = []

    OUTER_FOLDS = 5
    OUTER_REPEATS = 3  # 5x3 = 15 evaluations per section
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_FOLDS,
        n_repeats=OUTER_REPEATS,
        random_state=42
    )
    N_JOBS_MODELS = -1  # -1 means use all available cores

    print(f"\nUsing RepeatedStratifiedKFold: "
        f"{OUTER_FOLDS} folds × {OUTER_REPEATS} repeats")
    print("=" * 70)

    for criteria_name, threshold in criteria.items():
        print(f"\n{'='*70}")
        print(f"CRITERIA: {criteria_name}")
        print('='*70)

        filtered = tc.filter_dataset(df, criteria_name, threshold)

        for section in sections:
            print(f"\n{'-'*70}")
            print(f"Section: {section}")
            print('-'*70)

            section_df = filtered[filtered['section'] == section]

            if len(section_df) < 20:
                print("⚠ Not enough samples. Skipping.")
                continue

            X = section_df[all_features]
            y = section_df["interacting_complex"]

            if y.nunique() < 2:
                print("⚠ Only one class present. Skipping.")
                continue

            print(f"Total samples: {len(y)}")
            print(f"Positive %: {y.mean()*100:.2f}%")

            section_outer_results = []

            # =====================================================
            # OUTER CV LOOP (variance reduction)
            # =====================================================
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Safeguard for extreme imbalance
                if y_train.sum() < 10 or y_test.sum() < 5:
                    print(f"  Fold {fold_idx}: skipped (too few positives)")
                    continue

                strategy, ratio, reason = tc.get_balancing_strategy(y_train, section)
                base_models = tc.build_base_models(
                    imbalance_ratio=ratio if ratio else 1.0,
                    strategy=strategy
                )

                # =================================================
                # PARALLEL MODEL TRAINING (same as before)
                # Inner CV threshold tuning happens inside
                # train_single_model (unchanged)
                # =================================================
                model_results = Parallel(n_jobs=N_JOBS_MODELS)(
                    delayed(tc.train_single_model)(
                        name, base_model,
                        X_train, y_train,
                        X_test, y_test,
                        strategy, section, criteria_name,
                        y_train.value_counts(),
                        section_df
                    )
                    for name, base_model in base_models.items()
                )

                for result in model_results:
                    if result is None:
                        continue
                    result["Outer Fold"] = fold_idx
                    section_outer_results.append(result)

            # =====================================================
            # SECTION-LEVEL AGGREGATION
            # =====================================================
            if section_outer_results:
                section_df_results = pd.DataFrame(section_outer_results)

                agg_section = (
                    section_df_results
                    .groupby("Model")
                    .agg({
                        "Average Precision": ["mean", "std"],
                        "F1 (Optimal)": ["mean", "std"],
                        "AUC-ROC": ["mean", "std"]
                    })
                    .reset_index()
                )

                agg_section.columns = [
                    "Model",
                    "AP_mean", "AP_std",
                    "F1_mean", "F1_std",
                    "AUC_mean", "AUC_std"
                ]

                agg_section["Section"] = section
                agg_section["Criteria"] = criteria_name

                results.append(agg_section)

    # ============================================================
    # CROSS-SECTION AGGREGATION
    # ============================================================

    if results:

        all_sections_df = pd.concat(results, ignore_index=True)

        print("\n" + "="*70)
        print("PER-SECTION RESULTS")
        print("="*70)
        print(all_sections_df.sort_values(
            ["Section", "AP_mean"], ascending=[True, False]
        ).to_string(index=False))

        # --------------------------------------------------------
        # Rank models within each section (based on AP_mean)
        # --------------------------------------------------------
        all_sections_df["Rank"] = (
            all_sections_df
            .groupby(["Section", "Criteria"])["AP_mean"]
            .rank(ascending=False)
        )

        # --------------------------------------------------------
        # Aggregate ranks across sections
        # --------------------------------------------------------
        rank_summary = (
            all_sections_df
            .groupby("Model")
            .agg({
                "Rank": ["mean"],
                "AP_mean": ["mean"],
                "AP_std": ["mean"]
            })
            .reset_index()
        )

        rank_summary.columns = [
            "Model",
            "Mean Rank",
            "Mean AP Across Sections",
            "Mean AP Std Across Sections"
        ]

        print("\n" + "="*70)
        print("CROSS-SECTION MODEL SELECTION SUMMARY")
        print("="*70)
        print(rank_summary.sort_values(
            "Mean Rank"
        ).to_string(index=False))

        print("\n🏆 Recommended Model:")
        best_model = rank_summary.sort_values("Mean Rank").iloc[0]["Model"]
        print(f"→ {best_model}")

        all_sections_df.to_csv("repeated_stratified_outer_cv.tsv", sep='\t', index=False)