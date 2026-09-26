def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    if getattr(args, 'data', '') == 'pv_multires':
        print(f'  {"Data:":<20}{args.data:<20}{"Features:":<20}{args.features:<20}')
    else:
        print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
        print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    if hasattr(args, 'manifest_path'):
        print(f'  {"Manifest Path:":<20}{args.manifest_path:<20}')
    if hasattr(args, 'station_data_root'):
        print(f'  {"Station Root:":<20}{args.station_data_root:<20}')
    if hasattr(args, 'region_filter'):
        print(f'  {"Region Filter:":<20}{args.region_filter:<20}{"Granularity:":<20}{args.granularity_filter:<20}')
    if hasattr(args, 'source_region_filter'):
        print(f'  {"Source Region:":<20}{args.source_region_filter:<20}')
    if hasattr(args, 'station_dir_filter'):
        print(f'  {"Station Filter:":<20}{args.station_dir_filter:<20}{"Max Stations:":<20}{args.max_stations:<20}')
    if hasattr(args, 'data_file_name'):
        print(f'  {"Data File:":<20}{args.data_file_name:<20}{"NWP Mode:":<20}{args.nwp_mode:<20}')
    if hasattr(args, 'resample_to_1h'):
        print(f'  {"Resample 1h:":<20}{str(args.resample_to_1h):<20}{"Weather File:":<20}{getattr(args, "weather_file_name", ""):<20}')
    if hasattr(args, 'history_covariate_file_name'):
        print(f'  {"Hist Cov File:":<20}{getattr(args, "history_covariate_file_name", ""):<20}')
    if hasattr(args, 'history_covariate_cols'):
        print(f'  {"Hist Cov Cols:":<20}{getattr(args, "history_covariate_cols", ""):<20}')
    if hasattr(args, 'future_covariate_file_name'):
        print(f'  {"Future Cov File:":<20}{getattr(args, "future_covariate_file_name", ""):<20}')
    if hasattr(args, 'future_covariate_cols'):
        print(f'  {"Future Cov Cols:":<20}{getattr(args, "future_covariate_cols", ""):<20}')
    if hasattr(args, 'future_covariate_align_to_hour'):
        print(f'  {"Future Align:":<20}{getattr(args, "future_covariate_align_to_hour", ""):<20}')
    if hasattr(args, 'enable_physical_sample_filter'):
        print(f'  {"Phys Filter:":<20}{str(getattr(args, "enable_physical_sample_filter", False)):<20}')
        print(f'  {"Hist Solar Col:":<20}{getattr(args, "physical_filter_history_solar_col", ""):<20}')
        print(f'  {"Fut Solar Col:":<20}{getattr(args, "physical_filter_future_solar_col", ""):<20}')
        print(f'  {"Daylight Thr:":<20}{str(getattr(args, "physical_filter_daylight_threshold", "")):<20}{"Zero Thr:":<20}{str(getattr(args, "physical_filter_target_zero_threshold", "")):<20}')
        print(f'  {"Max Fut Zero:":<20}{str(getattr(args, "physical_filter_max_future_day_zero_ratio", "")):<20}{"Max Hist Zero:":<20}{str(getattr(args, "physical_filter_max_history_day_zero_ratio", "")):<20}')
        print(f'  {"Min Hist Corr:":<20}{str(getattr(args, "physical_filter_min_history_day_corr", "")):<20}{"Min Fut DayPts:":<20}{str(getattr(args, "physical_filter_min_future_daylight_points", "")):<20}')
    print()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        print(f'  {"Inverse:":<20}{args.inverse:<20}')
        print()

    if args.task_name == 'imputation':
        print("\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        print()

    if args.task_name == 'anomaly_detection':
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    print(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    print(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    print(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    print(f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    print(f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    if hasattr(args, 'fusionsf_ctx_source'):
        print(f'  {"FSF Ctx Src:":<20}{args.fusionsf_ctx_source:<20}{"FSF Dim:":<20}{args.fusionsf_dim:<20}')
        print(f'  {"FSF Depth:":<20}{args.fusionsf_depth:<20}{"FSF Heads:":<20}{args.fusionsf_heads:<20}')
        print(f'  {"FSF Dropout:":<20}{args.fusionsf_dropout:<20}{"FSF FF Mult:":<20}{args.fusionsf_ff_mult:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print(f'  {"Seed:":<20}{getattr(args, "seed", "unset"):<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}') 
    print()
