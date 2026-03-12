def set_gravnet_hls_config(hls_config: dict):
    hls_config['Model']['Precision'] = {'default': 'ap_fixed<16,6>', 'maximum': 'ap_fixed<18,8>'}
    hls_config['Model']['Strategy'] = 'Latency'

    for layer in hls_config['LayerName'].keys():
        if 'core' in layer:
            hls_config['LayerName'][layer]['ExponentialTable'] = {
                'ScaleFactor': 2,
                'Resolution': 64,
            }
            hls_config['LayerName'][layer]['Precision']['coords_diff'] = 'ap_fixed<8, 3>'
            hls_config['LayerName'][layer]['Precision']['exp_table'] = 'ap_ufixed<8, 1>'

        if 'gex' in layer:
            hls_config['LayerName'][layer]['Precision']['mean'] = 'ap_fixed<22,12>'

        layer_precision_keys = hls_config['LayerName'][layer]['Precision'].keys()
        if 'weight' in layer_precision_keys:
            hls_config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<8,1,AP_RND,AP_SAT>'
        if 'bias' in layer_precision_keys:
            hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<8,1,AP_RND,AP_SAT>'
        if 'accum' in layer_precision_keys:
            hls_config['LayerName'][layer]['Precision']['accum'] = 'ap_fixed<20,8>'

        hls_config['LayerName']['global_avg_pool']['Precision']['accum'] = 'ap_fixed<22,16>'


def set_converter_opts(converter_opts: dict, backend: str = 'Vitis'):
    converter_opts['io_type'] = 'io_stream'
    if backend == 'Vitis':
        pass  # Leave defaults
    elif backend == 'CoyoteAccelerator':
        converter_opts['clock_period'] = 4


def get_build_opts(backend: str = 'Vitis') -> dict:
    build_opts = {
        'reset': True,
        'csim': True,
        'synth': True,
        'cosim': True,
        'validation': True,
    }
    if backend == 'CoyoteAccelerator':
        build_opts['hls_clock_period'] = 5
        build_opts['csynth'] = True
        build_opts['timing_opt'] = True
        build_opts['bitfile'] = True
    else:
        build_opts['vsynth'] = True

    return build_opts
