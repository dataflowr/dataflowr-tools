"""
Full dataflowr course catalog.
Ground truth: https://dataflowr.github.io/website/
Notebooks:    https://github.com/dataflowr/notebooks
"""

from .models import Course, Module, Session, Homework, Notebook, NotebookKind

GITHUB_BASE = "https://github.com/dataflowr/notebooks/blob/master"
COLAB_BASE  = "https://colab.research.google.com/github/dataflowr/notebooks/blob/master"
WEBSITE_BASE = "https://dataflowr.github.io/website"
SLIDES_BASE  = "https://dataflowr.github.io/slides"

FLASH_GITHUB = "https://github.com/dataflowr/gpu_llm_flash-attention"
FLASH_COLAB  = "https://colab.research.google.com/github/dataflowr/gpu_llm_flash-attention/blob/main"

QUIZ_GITHUB = "https://github.com/dataflowr/quiz"

LLM_GEN_GITHUB = "https://github.com/dataflowr/llm_controlled-generation"

LLM_EFF_GITHUB = "https://github.com/dataflowr/llm_efficiency"
LLM_EFF_COLAB  = "https://colab.research.google.com/github/dataflowr/llm_efficiency/blob/main"


def _nb(filename: str, title: str, kind: NotebookKind = NotebookKind.practical,
        colab: bool = True, gpu: bool = False) -> Notebook:
    return Notebook(
        filename=filename,
        title=title,
        kind=kind,
        github_url=f"{GITHUB_BASE}/{filename}",
        colab_url=f"{COLAB_BASE}/{filename}" if colab else None,
        requires_gpu=gpu,
    )


def _flash_nb(path: str, title: str, kind: NotebookKind = NotebookKind.practical) -> Notebook:
    """Notebook entry for the gpu_llm_flash-attention repo."""
    is_ipynb = path.endswith(".ipynb")
    return Notebook(
        filename=path.split("/")[-1],
        title=title,
        kind=kind,
        github_url=f"{FLASH_GITHUB}/blob/main/{path}",
        colab_url=f"{FLASH_COLAB}/{path}" if is_ipynb else None,
        requires_gpu=True,
    )


def _llm_gen_nb(path: str, title: str, kind: NotebookKind = NotebookKind.practical) -> Notebook:
    """Notebook entry for the llm_controlled-generation repo."""
    return Notebook(
        filename=path.split("/")[-1],
        title=title,
        kind=kind,
        github_url=f"{LLM_GEN_GITHUB}/blob/main/{path}",
        colab_url=None,
        requires_gpu=False,
    )


def _llm_eff_nb(path: str, title: str, kind: NotebookKind = NotebookKind.practical,
                gpu: bool = True) -> Notebook:
    """Notebook entry for the llm_efficiency repo."""
    is_ipynb = path.endswith(".ipynb")
    return Notebook(
        filename=path.split("/")[-1],
        title=title,
        kind=kind,
        github_url=f"{LLM_EFF_GITHUB}/blob/main/{path}",
        colab_url=f"{LLM_EFF_COLAB}/{path}" if is_ipynb else None,
        requires_gpu=gpu,
    )


MODULES: dict[str, Module] = {

    "1": Module(
        id="1",
        title="Introduction & General Overview",
        description="Finetune a pretrained VGG network for dogs vs cats classification. "
                    "First contact with PyTorch, dataloaders, transfer learning.",
        session=1,
        website_url=f"{WEBSITE_BASE}/modules/1-intro-general-overview",
        slides_url=f"{SLIDES_BASE}/module1.html",
        tags=["transfer learning", "VGG", "classification", "finetuning"],
        requires_gpu=True,
        notebooks=[
            _nb("Module1/01_intro.ipynb", "Finetuning VGG for dogs vs cats",
                kind=NotebookKind.intro, gpu=True),
            _nb("Module1/01_practical_empty.ipynb", "More dogs and cats (empty)",
                kind=NotebookKind.practical),
            _nb("Module1/sol/01_practical_sol.ipynb", "More dogs and cats (solution)",
                kind=NotebookKind.solution),
        ],
    ),

    "2a": Module(
        id="2a",
        title="PyTorch Tensors",
        description="PyTorch tensors as numpy on GPU. Broadcasting, indexing, "
                    "in-place operations.",
        session=2,
        website_url=f"{WEBSITE_BASE}/modules/2a-pytorch-tensors",
        tags=["pytorch", "tensors", "numpy", "broadcasting"],
        quiz_files=["quiz_21.toml", "quiz_22.toml"],
        notebooks=[
            _nb("Module2/02a_basics.ipynb", "Basics on PyTorch tensors",
                kind=NotebookKind.intro),
            _nb("Module2/02b_linear_reg.ipynb", "Linear regression: numpy to pytorch",
                kind=NotebookKind.intro),
        ],
    ),

    "2b": Module(
        id="2b",
        title="Automatic Differentiation",
        description="Backpropagation as an algorithm, not just the chain rule. "
                    "Implement backprop from scratch.",
        session=2,
        prerequisites=["2a"],
        website_url=f"{WEBSITE_BASE}/modules/2b-automatic-differentiation",
        tags=["autodiff", "backpropagation", "gradients", "computational graph"],
        quiz_files=["quiz_23.toml"],
        notebooks=[
            _nb("Module2/02_backprop.ipynb", "Backprop from scratch (empty)",
                kind=NotebookKind.practical),
            _nb("Module2/sol/02_backprop_sol.ipynb", "Backprop from scratch (solution)",
                kind=NotebookKind.solution),
            _nb("Module2/autodiff_functional_empty.ipynb", "Autodiff the functional way (JAX)",
                kind=NotebookKind.bonus),
            _nb("Module2/linear_regression_jax.ipynb", "Linear regression in JAX",
                kind=NotebookKind.bonus),
            _nb("Module2/AD_with_dual_numbers_Julia.ipynb",
                "Autodiff with dual numbers (Julia)", kind=NotebookKind.bonus),
        ],
    ),

    "2c": Module(
        id="2c",
        title="Automatic Differentiation: VJP and intro to JAX",
        description="Vector-Jacobian products, functional autodiff, introduction to JAX.",
        session=2,
        prerequisites=["2b"],
        website_url=f"{WEBSITE_BASE}/modules/2c-jax",
        tags=["JAX", "VJP", "autodiff", "functional"],
        notebooks=[
            _nb("Module2/autodiff_functional_empty.ipynb",
                "Autodiff the functional way (empty)", kind=NotebookKind.practical),
            _nb("Module2/autodiff_functional_sol.ipynb",
                "Autodiff the functional way (solution)", kind=NotebookKind.solution),
        ],
    ),

    "3": Module(
        id="3",
        title="Loss Functions for Classification",
        description="Cross-entropy, softmax, binary vs multiclass. "
                    "Overfitting and underfitting via polynomial regression.",
        session=3,
        prerequisites=["2a", "2b"],
        website_url=f"{WEBSITE_BASE}/modules/3-loss-functions-for-classification",
        slides_url=f"{SLIDES_BASE}/module3.html",
        tags=["loss", "cross-entropy", "softmax", "classification", "overfitting"],
        quiz_files=[
            "quiz_31.toml", "quiz_32.toml", "quiz_33.toml", "quiz_34.toml",
            "quiz_35.toml", "quiz_36.toml", "quiz_37.toml", "quiz_38.toml",
        ],
        notebooks=[
            _nb("Module3/03_polynomial_regression.ipynb",
                "Overfitting/underfitting with polynomial regression",
                kind=NotebookKind.intro),
        ],
    ),

    "4": Module(
        id="4",
        title="Optimization for Deep Learning",
        description="SGD, momentum, Adagrad, RMSProp, Adam, AMSGrad. "
                    "Learning rate schedules.",
        session=3,
        prerequisites=["3", "2b"],
        website_url=f"{WEBSITE_BASE}/modules/4-optimization-for-deep-learning",
        slides_url=f"{SLIDES_BASE}/module4.html",
        tags=["optimization", "SGD", "Adam", "learning rate", "momentum"],
        notebooks=[
            _nb("Module4/04_gradient_descent_optimization_algorithms_empty.ipynb",
                "Implement Adagrad, RMSProp, Adam, AMSGrad (empty)",
                kind=NotebookKind.practical),
            _nb("Module4/sol/04_gradient_descent_optimization_algorithms_sol.ipynb",
                "Implement optimizers (solution)", kind=NotebookKind.solution),
        ],
    ),

    "5": Module(
        id="5",
        title="Stacking Layers",
        description="Building neural networks with torch.nn.Module. "
                    "Overfitting a MLP on CIFAR10.",
        session=3,
        prerequisites=["4", "3"],
        website_url=f"{WEBSITE_BASE}/modules/5-stacking-layers",
        slides_url=f"{SLIDES_BASE}/module5.html",
        tags=["MLP", "nn.Module", "CIFAR10", "layers"],
        notebooks=[
            _nb("Module5/Stacking_layers_MLP_CIFAR10.ipynb",
                "Overfitting a MLP on CIFAR10 (empty)", kind=NotebookKind.practical),
            _nb("Module5/sol/MLP_CIFAR10.ipynb",
                "MLP on CIFAR10 (solution)", kind=NotebookKind.solution),
        ],
    ),

    "6": Module(
        id="6",
        title="Convolutional Neural Networks",
        description="Convolution, pooling, stride, padding. "
                    "Build a digit recognizer with CNN.",
        session=3,
        prerequisites=["5"],
        website_url=f"{WEBSITE_BASE}/modules/6-convolutional-neural-network",
        tags=["CNN", "convolution", "pooling", "MNIST", "image classification"],
        notebooks=[
            _nb("Module6/06_convolution_digit_recognizer.ipynb",
                "Digit recognizer with CNN", kind=NotebookKind.practical),
        ],
    ),

    "7": Module(
        id="7",
        title="Dataloading",
        description="PyTorch Dataset and DataLoader API. "
                    "Custom datasets, transforms, batching.",
        session=4,
        prerequisites=["5"],
        website_url=f"{WEBSITE_BASE}/modules/7-dataloading",
        slides_url=f"{SLIDES_BASE}/module7.html",
        tags=["dataloader", "dataset", "transforms", "batching"],
        notebooks=[],
    ),

    "8a": Module(
        id="8a",
        title="Embedding Layers",
        description="Representing categorical variables as dense vectors. "
                    "Embedding tables in PyTorch.",
        session=4,
        prerequisites=["5"],
        website_url=f"{WEBSITE_BASE}/modules/8a-embedding-layers",
        slides_url=f"{SLIDES_BASE}/module8a.html",
        tags=["embeddings", "categorical", "representation"],
        notebooks=[],
    ),

    "8b": Module(
        id="8b",
        title="Collaborative Filtering",
        description="Matrix factorization with embeddings. "
                    "Build a recommender system on MovieLens.",
        session=4,
        prerequisites=["8a"],
        website_url=f"{WEBSITE_BASE}/modules/8b-collaborative-filtering",
        tags=["collaborative filtering", "recommender systems", "matrix factorization",
              "MovieLens"],
        notebooks=[
            _nb("Module8/08_collaborative_filtering_empty.ipynb",
                "Collaborative filtering on MovieLens 100k (empty)",
                kind=NotebookKind.practical),
            _nb("Module8/08_collaborative_filtering_1M.ipynb",
                "Collaborative filtering on MovieLens 1M",
                kind=NotebookKind.practical),
        ],
    ),

    "8c": Module(
        id="8c",
        title="Word2Vec",
        description="Word embeddings via skip-gram and negative sampling. "
                    "Synonyms and analogies with GloVe.",
        session=4,
        prerequisites=["8a"],
        website_url=f"{WEBSITE_BASE}/modules/8c-word2vec",
        tags=["word2vec", "word embeddings", "skip-gram", "negative sampling", "NLP"],
        notebooks=[
            _nb("Module8/08_Word2vec_pytorch_empty.ipynb",
                "Word2Vec in PyTorch (empty)", kind=NotebookKind.practical),
            _nb("Module8/08_Playing_with_word_embedding.ipynb",
                "Synonyms and analogies with GloVe", kind=NotebookKind.intro),
        ],
    ),

    "9a": Module(
        id="9a",
        title="Autoencoders",
        description="Encoder-decoder architecture. Denoising autoencoders "
                    "with transposed convolutions.",
        session=5,
        prerequisites=["5", "6"],
        website_url=f"{WEBSITE_BASE}/modules/9a-autoencoders",
        slides_url=f"{SLIDES_BASE}/module9.html",
        tags=["autoencoder", "encoder", "decoder", "transposed convolution",
              "denoising", "unsupervised"],
        notebooks=[
            _nb("Module9/09_AE_NoisyAE.ipynb",
                "Denoising autoencoder (conv + transposed conv)",
                kind=NotebookKind.practical),
        ],
    ),

    "9b": Module(
        id="9b",
        title="UNets",
        description="U-Net architecture for image segmentation. "
                    "Skip connections across encoder and decoder.",
        session=8,
        prerequisites=["9a", "6"],
        website_url=f"{WEBSITE_BASE}/modules/9b-unet",
        tags=["UNet", "segmentation", "skip connections", "encoder-decoder"],
        notebooks=[
            _nb("Module9/UNet_image_seg.ipynb",
                "UNet for image segmentation", kind=NotebookKind.practical),
        ],
    ),

    "9c": Module(
        id="9c",
        title="Normalizing Flows",
        description="Change of variables, Jacobian determinants. "
                    "Implement Real NVP from scratch.",
        session=8,
        prerequisites=["9a"],
        website_url=f"{WEBSITE_BASE}/modules/9c-flows",
        tags=["normalizing flows", "Real NVP", "generative models",
              "change of variables", "bijection"],
        notebooks=[
            _nb("Module9/Normalizing_flows_empty.ipynb",
                "Real NVP (empty)", kind=NotebookKind.practical),
            _nb("Module9/Normalizing_flows_sol.ipynb",
                "Real NVP (solution)", kind=NotebookKind.solution),
        ],
    ),

    "10": Module(
        id="10",
        title="Generative Adversarial Networks",
        description="GAN training dynamics, mode collapse. "
                    "Conditional GAN and InfoGAN on double moon dataset.",
        session=5,
        prerequisites=["9a"],
        website_url=f"{WEBSITE_BASE}/modules/10-generative-adversarial-networks",
        slides_url=f"{SLIDES_BASE}/module10.html",
        tags=["GAN", "generative models", "conditional GAN", "InfoGAN",
              "adversarial training"],
        notebooks=[
            _nb("Module10/10_GAN_double_moon.ipynb",
                "GAN, Conditional GAN, InfoGAN on double moon",
                kind=NotebookKind.practical),
        ],
    ),

    "11a": Module(
        id="11a",
        title="Recurrent Neural Networks (Theory)",
        description="Vanishing gradients, BPTT, LSTM, GRU. "
                    "The theory behind sequence modeling.",
        session=6,
        prerequisites=["5"],
        website_url=f"{WEBSITE_BASE}/modules/11a-recurrent-neural-networks-theory",
        slides_url=f"{SLIDES_BASE}/module11.html",
        tags=["RNN", "LSTM", "GRU", "BPTT", "vanishing gradients", "sequences"],
        notebooks=[
            _nb("Module11/11_RNN.ipynb", "RNN theory notebook",
                kind=NotebookKind.intro),
        ],
    ),

    "11b": Module(
        id="11b",
        title="Recurrent Neural Networks (Practice)",
        description="RNNs in PyTorch. Predict engine failure from time series.",
        session=6,
        prerequisites=["11a"],
        website_url=f"{WEBSITE_BASE}/modules/11b-recurrent-neural-networks-practice",
        slides_url=f"{SLIDES_BASE}/module11.html",
        tags=["RNN", "LSTM", "time series", "sequence modeling"],
        notebooks=[
            _nb("Module11/11_predictions_RNN_empty.ipynb",
                "Predicting engine failure with RNN (empty)",
                kind=NotebookKind.practical),
        ],
    ),

    "11c": Module(
        id="11c",
        title="Batches with Sequences in PyTorch",
        description="Padding, packing, masking. Handling variable-length sequences.",
        session=6,
        prerequisites=["11b"],
        website_url=f"{WEBSITE_BASE}/modules/11c-batches-with-sequences",
        tags=["sequences", "padding", "packing", "variable length", "collate"],
        notebooks=[],
    ),

    "12": Module(
        id="12",
        title="Attention and Transformers",
        description="Self-attention, multi-head attention, positional encoding. "
                    "Build microGPT from scratch.",
        session=7,
        prerequisites=["11a", "8a"],
        website_url=f"{WEBSITE_BASE}/modules/12-attention",
        tags=["attention", "transformer", "self-attention", "GPT", "seq2seq",
              "positional encoding"],
        notebooks=[
            _nb("Module12/12_seq2seq_attention.ipynb",
                "Corrected seq2seq with attention (empty)",
                kind=NotebookKind.practical),
            _nb("Module12/12_seq2seq_attention_solution.ipynb",
                "seq2seq with attention (solution)", kind=NotebookKind.solution),
            _nb("Module12/GPT_hist.ipynb",
                "Build your own microGPT (empty)", kind=NotebookKind.practical),
            _nb("Module12/GPT_hist_sol.ipynb",
                "microGPT (solution)", kind=NotebookKind.solution),
        ],
    ),

    "13": Module(
        id="13",
        title="Siamese Networks and Representation Learning",
        description="Contrastive learning, triplet loss. "
                    "Learning embeddings on MNIST.",
        session=5,
        prerequisites=["9a", "6"],
        website_url=f"{WEBSITE_BASE}/modules/13-siamese",
        slides_url=f"{SLIDES_BASE}/13-siamese-networks.html",
        tags=["siamese networks", "contrastive loss", "triplet loss",
              "representation learning", "metric learning"],
        notebooks=[
            _nb("Module13/13_siamese_triplet_mnist_empty.ipynb",
                "Contrastive loss embeddings on MNIST (empty)",
                kind=NotebookKind.practical),
        ],
    ),

    "14a": Module(
        id="14a",
        title="The Benefits of Depth",
        description="Why depth helps: expressivity, compositionality, "
                    "function approximation theory.",
        session=4,
        prerequisites=["5"],
        website_url=f"{WEBSITE_BASE}/modules/14a-depth",
        slides_url=f"{SLIDES_BASE}/14-01-deeper.html",
        tags=["depth", "expressivity", "universal approximation", "theory"],
        notebooks=[],
    ),

    "14b": Module(
        id="14b",
        title="The Problems with Depth",
        description="Vanishing and exploding gradients, initialization strategies.",
        session=4,
        prerequisites=["14a"],
        website_url=f"{WEBSITE_BASE}/modules/14b-depth",
        slides_url=f"{SLIDES_BASE}/14-02-problems.html",
        tags=["vanishing gradients", "exploding gradients", "initialization", "depth"],
        notebooks=[],
    ),

    "15": Module(
        id="15",
        title="Dropout",
        description="Dropout as regularization and approximate Bayesian inference. "
                    "MC Dropout for uncertainty estimation.",
        session=3,
        prerequisites=["5"],
        website_url=f"{WEBSITE_BASE}/modules/15-dropout",
        slides_url=f"{SLIDES_BASE}/14-03-dropout.html",
        tags=["dropout", "regularization", "uncertainty", "MC dropout", "Bayesian"],
        notebooks=[
            _nb("Module15/15a_dropout_intro.ipynb",
                "Dropout on toy dataset", kind=NotebookKind.intro),
            _nb("Module15/15b_dropout_mnist.ipynb",
                "Dropout on MNIST", kind=NotebookKind.practical),
        ],
    ),

    "16": Module(
        id="16",
        title="Batch Normalization",
        description="BatchNorm mechanics, train vs eval behavior, "
                    "why it helps optimization.",
        session=4,
        prerequisites=["5", "4"],
        website_url=f"{WEBSITE_BASE}/modules/16-batchnorm",
        slides_url=f"{SLIDES_BASE}/14-04-batchnorm.html",
        tags=["batchnorm", "normalization", "train/eval", "internal covariate shift"],
        notebooks=[
            _nb("Module16/16_batchnorm_simple.ipynb",
                "Impact of batchnorm", kind=NotebookKind.intro),
            _nb("Module16/16_simple_batchnorm_eval.ipynb",
                "Batchnorm without training", kind=NotebookKind.practical),
        ],
    ),

    "17": Module(
        id="17",
        title="ResNets",
        description="Residual connections, skip connections, "
                    "out-of-distribution detection with ODIN.",
        session=4,
        prerequisites=["14b", "16"],
        website_url=f"{WEBSITE_BASE}/modules/17-resnets",
        slides_url=f"{SLIDES_BASE}/14-05-resnets.html",
        tags=["ResNet", "residual connections", "skip connections",
              "OOD detection", "ODIN"],
        notebooks=[
            _nb("Module17/ODIN_mobilenet_empty.ipynb",
                "OOD detection with ODIN (empty)", kind=NotebookKind.practical),
        ],
    ),

    "18a": Module(
        id="18a",
        title="Denoising Score Matching for Energy Based Models",
        description="Energy-based models, score matching, "
                    "denoising score matching, Langevin dynamics.",
        session=9,
        prerequisites=["9a", "4"],
        website_url=f"{WEBSITE_BASE}/modules/18a-energy",
        tags=["energy-based models", "score matching", "EBM", "Langevin dynamics",
              "denoising"],
        notebooks=[],
    ),

    "18b": Module(
        id="18b",
        title="Denoising Diffusion Probabilistic Models",
        description="DDPM: forward process, reverse process, noise prediction. "
                    "Train on MNIST and CIFAR10.",
        session=9,
        prerequisites=["18a"],
        website_url=f"{WEBSITE_BASE}/modules/18b-diffusion",
        tags=["diffusion models", "DDPM", "score matching", "generative models",
              "denoising"],
        requires_gpu=True,
        notebooks=[
            _nb("Module18/ddpm_nano_empty.ipynb",
                "DDPM on MNIST (empty)", kind=NotebookKind.practical, gpu=True),
            _nb("Module18/ddpm_nano_sol.ipynb",
                "DDPM on MNIST (solution)", kind=NotebookKind.solution, gpu=True),
            _nb("Module18/ddpm_micro_sol.ipynb",
                "DDPM finetuning on CIFAR10", kind=NotebookKind.solution, gpu=True),
        ],
    ),

    "19": Module(
        id="19",
        title="Zero-shot Classification with CLIP",
        description="Contrastive Language-Image Pre-Training. "
                    "Zero-shot transfer, multimodal embeddings.",
        session=9,
        prerequisites=["12", "13"],
        website_url=f"{WEBSITE_BASE}/modules/19-clip",
        tags=["CLIP", "zero-shot", "multimodal", "contrastive learning",
              "vision-language"],
        requires_gpu=True,
        notebooks=[],
    ),

    "flash": Module(
        id="flash",
        title="Flash Attention in Triton",
        description="Implement FlashAttention-2 from scratch using Triton. "
                    "Online softmax, memory-efficient attention, GPU kernel optimization.",
        prerequisites=["12"],
        website_url=FLASH_GITHUB,
        tags=["flash attention", "Triton", "GPU kernels", "attention", "optimization", "CUDA"],
        requires_gpu=True,
        notebooks=[
            _flash_nb("FlashAttention_empty.ipynb", "Flash Attention in Triton (empty)"),
            _flash_nb("homework/01_softmax_matmul.md",
                      "Task 1: Softmax-Matmul Kernel", kind=NotebookKind.homework),
            _flash_nb("homework/02_flash_attention_pytorch.md",
                      "Task 2: Flash Attention in PyTorch", kind=NotebookKind.homework),
            _flash_nb("homework/03_flash_attention_triton.md",
                      "Task 3: Flash Attention in Triton", kind=NotebookKind.homework),
        ],
    ),

    "llm_gen": Module(
        id="llm_gen",
        title="LLM Controlled Generation",
        description="Three-part lab on controlling LLM outputs: FSM-based constrained decoding "
                    "for structured generation, best-of-N meta-generation with MBR selection, "
                    "and tree-search self-correction for formally verified Rust code.",
        prerequisites=["12"],
        website_url=LLM_GEN_GITHUB,
        tags=["LLM", "structured generation", "constrained decoding", "FSM",
              "meta-generation", "MBR", "self-correction", "code generation"],
        notebooks=[
            _llm_gen_nb("structured_generation/structured_generation.md",
                        "Part 1: Structured Generation (FSM constrained decoding)"),
            _llm_gen_nb("meta_generation/meta_generation.md",
                        "Part 2: Meta-Generation (best-of-N with MBR)"),
            _llm_gen_nb("self_correction/self_correction.md",
                        "Part 3: Self-Correction (tree-search + Verus)"),
        ],
    ),

    "kv_cache": Module(
        id="kv_cache",
        title="KV Cache for Efficient LLM Inference",
        description="Implement KV caching for transformer inference to avoid redundant "
                    "key/value recomputation. Benchmark the speedup on a minGPT model.",
        prerequisites=["12"],
        website_url=LLM_EFF_GITHUB,
        tags=["KV cache", "inference", "minGPT", "optimization", "LLM", "transformer"],
        requires_gpu=True,
        notebooks=[
            _llm_eff_nb("kv_cache/kv_cache.md",
                        "KV Cache: Concept and API", kind=NotebookKind.intro, gpu=False),
            _llm_eff_nb("practicals/KV_cache_empty.ipynb",
                        "KV Cache Implementation (empty)"),
        ],
    ),

    "lora": Module(
        id="lora",
        title="LoRA: Low-Rank Adaptation for LLMs",
        description="Implement LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. "
                    "Apply it to a minGPT sorting task.",
        prerequisites=["12"],
        website_url=LLM_EFF_GITHUB,
        tags=["LoRA", "fine-tuning", "parameter-efficient", "minGPT", "LLM", "low-rank"],
        requires_gpu=True,
        notebooks=[
            _llm_eff_nb("lora/lora.md",
                        "LoRA: Concept and API", kind=NotebookKind.intro, gpu=False),
            _llm_eff_nb("practicals/Lora_empty.ipynb",
                        "LoRA Fine-Tuning (empty)"),
        ],
    ),

    "graph0": Module(
        id="graph0",
        title="Deep Learning on Graphs",
        description="Graph neural networks, message passing, GCN. "
                    "Spectral perspective on graph convolutions.",
        prerequisites=["5", "6"],
        website_url=f"{WEBSITE_BASE}/modules/graph0",
        slides_url=f"{SLIDES_BASE}/deep_graph_0.html",
        tags=["GNN", "graph neural networks", "GCN", "message passing",
              "graph convolution"],
        notebooks=[
            _nb("graphs/GCN_inductivebias_spectral.ipynb",
                "Inductive bias in GCN: spectral perspective",
                kind=NotebookKind.intro),
            _nb("graphs/spectral_gnn.ipynb",
                "Graph ConvNets in PyTorch", kind=NotebookKind.practical),
        ],
    ),
}


SESSIONS: list[Session] = [
    Session(
        number=1,
        title="Finetuning VGG",
        modules=["1"],
        things_to_remember=[
            "You don't need to understand everything to run a deep learning model.",
            "To use the PyTorch dataloader for classification, store your dataset in folders.",
            "Using a pretrained model and modifying it for a similar task is easy.",
            "Even with a GPU, avoid unnecessary computations.",
        ],
    ),
    Session(
        number=2,
        title="PyTorch Tensors and Autodiff",
        modules=["2a", "2b", "2c"],
        things_to_remember=[
            "PyTorch tensors = Numpy on GPU + gradients.",
            "Broadcasting is used everywhere in deep learning — same rules as NumPy.",
            "Automatic differentiation is not only the chain rule! "
            "Backpropagation is a clever algorithm.",
        ],
    ),
    Session(
        number=3,
        title="Loss, Optimization, CNNs",
        modules=["3", "4", "5", "6", "15"],
        things_to_remember=[
            "Loss vs Accuracy: know your loss for a classification task.",
            "Know your optimizer.",
            "Know how to build a neural net with torch.nn.Module.",
            "Know how to use convolution and pooling layers (kernel, stride, padding).",
            "Know how to use dropout.",
        ],
    ),
    Session(
        number=4,
        title="Dataloading, Embeddings, Depth",
        modules=["7", "8a", "8b", "8c", "14a", "14b", "16", "17"],
        things_to_remember=[
            "Know how to use DataLoader.",
            "For categorical variables in deep learning, use embeddings.",
            "Word2Vec builds a supervised task from unsupervised data via negative sampling.",
            "Know your BatchNorm — especially train vs eval behavior.",
            "Skip connections allow training deeper models.",
        ],
    ),
    Session(
        number=5,
        title="Generative Models I: AE, GAN, Siamese",
        modules=["9a", "10", "13"],
        things_to_remember=[
            "Autoencoders learn compact representations by reconstruction.",
            "GANs are hard to train due to adversarial dynamics.",
            "Contrastive learning builds geometry in embedding space.",
        ],
    ),
    Session(
        number=6,
        title="Recurrent Neural Networks",
        modules=["11a", "11b", "11c"],
        things_to_remember=[
            "RNNs suffer from vanishing/exploding gradients.",
            "LSTMs and GRUs solve this with gating mechanisms.",
            "Variable-length sequences require careful batching (padding/packing).",
        ],
    ),
    Session(
        number=7,
        title="Attention and Transformers",
        modules=["12"],
        things_to_remember=[
            "Attention allows each token to look at all other tokens.",
            "Transformers replaced RNNs for most sequence tasks.",
            "Positional encoding is needed because attention is permutation-invariant.",
        ],
    ),
    Session(
        number=8,
        title="Generative Models II: UNets and Flows",
        modules=["9b", "9c"],
        things_to_remember=[
            "U-Nets use skip connections to preserve spatial information.",
            "Normalizing flows are bijective, enabling exact likelihood computation.",
            "Real NVP uses coupling layers to make the Jacobian tractable.",
        ],
    ),
    Session(
        number=9,
        title="Diffusion Models and CLIP",
        modules=["18a", "18b", "19"],
        things_to_remember=[
            "DDPMs learn to reverse a gradual noising process.",
            "Score matching connects energy-based models and diffusion.",
            "CLIP aligns image and text embeddings via contrastive training.",
        ],
    ),
]


HOMEWORKS: list[Homework] = [
    Homework(
        id=1,
        title="MLP from Scratch",
        description="Implement a multi-layer perceptron without using torch.nn, "
                    "relying only on tensor operations and autograd.",
        website_url=f"{WEBSITE_BASE}/homework/1-mlp-from-scratch",
        notebooks=[
            _nb("HW1/hw1_mlp.ipynb", "MLP from scratch (empty)",
                kind=NotebookKind.homework),
            _nb("HW1/sol/hw1_mlp_sol.ipynb", "MLP from scratch (solution)",
                kind=NotebookKind.solution),
        ],
    ),
    Homework(
        id=2,
        title="Class Activation Map and Adversarial Examples",
        description="Visualize what a CNN has learned via CAM. "
                    "Craft adversarial examples with FGSM.",
        website_url=f"{WEBSITE_BASE}/homework/2-CAM-adversarial",
        notebooks=[
            _nb("HW2/HW2_CAM_Adversarial.ipynb",
                "CAM and adversarial examples", kind=NotebookKind.homework),
        ],
    ),
    Homework(
        id=3,
        title="VAE for MNIST Clustering and Generation",
        description="Implement a Variational Autoencoder. "
                    "Use it to cluster and generate MNIST digits.",
        website_url=f"{WEBSITE_BASE}/homework/3-VAE",
        notebooks=[],
    ),
    Homework(
        id=4,
        title="Flash Attention Implementation",
        description="Implement FlashAttention-2 in PyTorch and Triton. "
                    "Three tasks: softmax-matmul kernel, PyTorch FA forward+backward, "
                    "Triton FA kernel.",
        website_url=FLASH_GITHUB,
        notebooks=[
            _flash_nb("FlashAttention_empty.ipynb", "Flash Attention Lab Notebook",
                      kind=NotebookKind.homework),
            _flash_nb("homework/01_softmax_matmul.md", "Task 1: Softmax-Matmul Kernel",
                      kind=NotebookKind.homework),
            _flash_nb("homework/02_flash_attention_pytorch.md",
                      "Task 2: Flash Attention in PyTorch", kind=NotebookKind.homework),
            _flash_nb("homework/03_flash_attention_triton.md",
                      "Task 3: Flash Attention in Triton", kind=NotebookKind.homework),
        ],
    ),
    Homework(
        id=5,
        title="LLM Controlled Generation",
        description="Three-part lab on controlling LLM outputs: implement FSM-based constrained "
                    "decoding for structured generation, best-of-N meta-generation with MBR "
                    "selection, and tree-search self-correction for formally verified Rust code.",
        website_url=LLM_GEN_GITHUB,
        notebooks=[
            _llm_gen_nb("structured_generation/structured_generation.md",
                        "Part 1: Structured Generation", kind=NotebookKind.homework),
            _llm_gen_nb("meta_generation/meta_generation.md",
                        "Part 2: Meta-Generation", kind=NotebookKind.homework),
            _llm_gen_nb("self_correction/self_correction.md",
                        "Part 3: Self-Correction", kind=NotebookKind.homework),
        ],
    ),

    Homework(
        id=6,
        title="LLM Efficiency: KV Cache and LoRA",
        description="Two-part lab on LLM efficiency: implement KV caching for fast "
                    "inference and LoRA for parameter-efficient fine-tuning, both "
                    "built on top of minGPT.",
        website_url=LLM_EFF_GITHUB,
        notebooks=[
            _llm_eff_nb("practicals/KV_cache_empty.ipynb",
                        "Part 1: KV Cache Implementation", kind=NotebookKind.homework),
            _llm_eff_nb("practicals/Lora_empty.ipynb",
                        "Part 2: LoRA Fine-Tuning", kind=NotebookKind.homework),
        ],
    ),
]


COURSE = Course(
    title="Deep Learning Do It Yourself!",
    description="A practical deep learning course focused on PyTorch from scratch. "
                "No high-level APIs. Students implement everything themselves, "
                "from backprop to diffusion models.",
    github_url="https://github.com/dataflowr/notebooks",
    website_url="https://dataflowr.github.io/website",
    modules=MODULES,
    sessions=SESSIONS,
    homeworks=HOMEWORKS,
)
