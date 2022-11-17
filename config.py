from pathlib import Path


class ModelConfig(object):
    root_dir = "/home/dp/Data/ClearDatasets/Datasets/aa-corners"
    negative_keys = "negative"
    positive_keys = "positive"
    positive_dir = "positive_full"
    features_dir = "features"
    pdb_base = "/home/dp/Data/PDB"
    models_dir = 'models'
    models_types = ['aa-corner', 'a-hairpin', 'b-hairpin', 'bab']
    out_dir = "/home/dp/Data/Extacted/"
    segmentation_model_path = "segmentation_model.pth"
    inference_model_path = "inference_model.pth"
    logfile = "metadata.csv"
    scheduler_type = None
    eval_models_every = 2
    n_base_processing_layers = 3
    n_post_processing_layers = 5
    inference_threshold = 0.95
    model_params = dict(
        inference_dim=256,
        self_att_dim=64,
        rnn_encoder_dim=16,
        rnn_decoder_dim=32,
        vx_input_ca=(6, 3),
        vx_input_c=(2, 2),
        vx_h_ca=(48, 32),
        vx_h_c=(32, 16),
        ex_input=(32, 1),
        ex_hidden=(64, 1),
        drop_rate=0.2,
        graph_attention=True
    )
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=True,
        threshold=0.001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-5,
        eps=1e-08
    )

    def __init__(self):
        pass

    def get_models_folder(self, model_type):
        assert model_type in self.models_types
        current_folder = Path(__file__).parent.absolute()
        model_folder = current_folder / self.models_dir / model_type
        return model_folder


