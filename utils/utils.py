# Load modules
import os, cv2, random, torch, logging
import numpy as np
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from rich.progress import (BarColumn, 
                           Progress,
                           TextColumn, 
                           TimeElapsedColumn, 
                           TimeRemainingColumn)



# Define the global logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Only add handler if none exists (avoid duplicates)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def flatten(input, target):
    """
        Flattens the input and target tensors for easier processing, commonly used in segmentation tasks.
        
        Args:
            input (torch.Tensor): The input tensor of shape (B, C, H, W).
            target (torch.Tensor): The target tensor of shape (B, C, H, W).
            
        Returns:
            input_flatten (torch.Tensor): Flattened input tensor of shape (B * H * W, C).
            target_flatten (torch.Tensor): Flattened target tensor of shape (B * H * W, C).
    """
    
    num_class = input.size(1)
    input = input.permute(0, 2, 3, 1).contiguous()
    
    input_flatten = input.view(-1, num_class)
    target_flatten = target.view(-1)
    
    return input_flatten, target_flatten


def set_reproducibility(seed: int, deterministic: bool) -> None:
    """
        Sets seeds and CUDA backend flags to ensure reproducible results in PyTorch experiments.
        
        Args:
            seed (int): The seed value for random number generators.
            deterministic (bool): If True, sets CUDA to deterministic mode for reproducibility,
                                otherwise enables benchmark mode for potentially better performance.
    """
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def worker_init_fn(worker_id: int, base_seed: int):
    """
        Creates a worker initialization function for PyTorch DataLoader workers to ensure
        reproducible random seeds across all workers.

        Each worker's random seed is set to `base_seed + worker_id` to avoid seed collisions
        and maintain reproducibility in multi-worker data loading.

        Args:
            worker_id (int): The worker ID used for the PyTorch DataLoader.
            base_seed (int): The base seed value used to derive each worker's seed.

        Returns:
            Callable: A function to be passed as `worker_init_fn` in DataLoader that sets the
                    random seed for each worker based on its worker ID.
    """
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)


def visualize_samples(dataloader, 
                      data_type: str,
                      num_samples: int = 4,
                      padding: int = 5,
                      pad_value: float = 0.5):
    """
    Visualizes a batch of images and corresponding masks from the dataloader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader yielding batches with keys 'image', 'label', 'mask' and 'plaque'.
        data_type (str): The type of the dataset used in the input dataloader.
        num_samples (int, optional): Number of samples to visualize from the (possibly) multiple batches. Defaults to 4.
        padding (int, optional): Width of padding in pixels. Defaults to 5.
        pad_value (float, optional): Value of padding (0=black, 1=white, 0.5=gray). Defaults to 0.5.

    Displays:
        All images stacked horizontally in one row,
        all masks stacked horizontally below the images.
    """

    # Helper function: stack horizontally with padding
    def add_horizontal_padding(list_of_arrays, padding, pad_value):
        padded_list = []
        for i, arr in enumerate(list_of_arrays):
            padded_list.append(arr)
            if i < len(list_of_arrays) - 1:
                pad_array = np.ones((arr.shape[0], padding)) * pad_value
                padded_list.append(pad_array)
        return np.hstack(padded_list)
    
    # Helper function: stack vertically with padding
    def add_vertical_padding(rows, padding, pad_value):
        padded_rows = []
        for i, row in enumerate(rows):
            padded_rows.append(row)
            if i < len(rows) - 1:
                pad_row = np.ones((padding, row.shape[1])) * pad_value
                padded_rows.append(pad_row)
        return np.vstack(padded_rows)
    
    imgs, names = [], []
    norm_lumens, norm_plaques = [], []

    for batch in dataloader:

        case_names = batch['case_name']
        raw_images = batch['raw_image']  
        lumens = batch['lumen']
        plaques = batch['plaque']   
        batch_size = raw_images.shape[0]

        for i in range(batch_size):
            
            img = raw_images[i].squeeze(0).cpu().numpy()
            lumen = lumens[i].cpu().numpy()
            plaque = plaques[i].cpu().numpy()
            case_name = case_names[i]

            if lumen.max() > 0:
                lumen = lumen / lumen.max()
            if plaque.max() > 0:
                plaque = plaque / plaque.max()
            
            imgs.append(img)
            names.append(case_name)
            norm_lumens.append(lumen)
            norm_plaques.append(plaque)

            if len(imgs) >= num_samples:
                break
        
        if len(imgs) >= num_samples:
            break

    # Just in case, limit to exactly num_samples
    imgs = imgs[:num_samples]
    names = names[:num_samples]
    norm_lumens = norm_lumens[:num_samples]
    norm_plaques = norm_plaques[:num_samples]

    # Stack horizontally
    row_images = add_horizontal_padding(imgs, padding, pad_value)
    row_lumens = add_horizontal_padding(norm_lumens, padding, pad_value)
    row_plaques = add_horizontal_padding(norm_plaques, padding, pad_value)

    # Stack images on top of masks vertically
    stacked = add_vertical_padding([row_images, row_lumens, row_plaques], padding, pad_value)

    # Compute positions of sample centers
    widths = [img.shape[1] for img in imgs]

    centers = []
    current_x = 0
    for w in widths:
        centers.append(current_x + w / 2)
        current_x += w + padding

    # Plot
    plt.figure(figsize = (15, 6))
    plt.imshow(stacked, cmap = 'gray')
    plt.axis('off')
    plt.title(f"{data_type.upper()}")

    # Add names above the first row
    y_text = 15 
    for center_x, name in zip(centers, names):
        plt.text(center_x, y_text, 
                 name,
                 fontsize = 7, 
                 color = 'white',
                 ha = 'center', 
                 va = 'bottom')

    plt.show()
    

def initialize_progress_bar() -> Progress:
    r""" 
        Initialize a general progress bar. 
        
        Returns:
            progress_bar (rich.progress.Progress): The instantiated progress bar.
    """
    progress =  Progress(
                            TextColumn("[progress.description]{task.description}"),
                            BarColumn(),
                            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                            TextColumn("•"),
                            "{task.completed}/{task.total}",
                            TextColumn("•"),
                            TimeElapsedColumn(),
                            TextColumn("•"),
                            TimeRemainingColumn()
                        )
    return progress


def find_file_in_subfolders(config_path: str, 
                            filename: str
                ) -> str:
    r""" 
        Return the subfolder of `config_path` containing the input `filename`.
        
        Args:
            config_path (str): The configuration folder.
            filename (str): The input yaml file.

        Returns:
            subfolder (str): The subfolder corresponding to the input filename. 
    """

    # Get the absolute path of the config_path
    config_path = os.path.abspath(config_path)
    
    # Loop through all subdirectories in config_path
    for subfolder, _, files in os.walk(config_path):

        # Check if the given filename (without extension) exists in the current folder
        for file in files:
           
            # Compare the filename without extension & return the subfolder name if found
            if os.path.splitext(file)[0] == filename:
                return os.path.basename(subfolder)
                
    # Raise an error if the file wasn't found in any subfolder
    raise FileNotFoundError(f"File `{filename}` not found in the subfolders of `{os.path.basename(config_path)}`.")  


def post_process_targets(lumen: torch.Tensor,
                         plaque: torch.Tensor) -> torch.Tensor:
    """
    Post-processes the lumen targets.

    Args:
        lumen (Tensor): binary region mask, shape (B, H, W)
        plaque (Tensor): binary plaque region, shape (B, H, W)

    Returns:
        Tensor: post-processed targets, shape (B, H, W)
    """

    # Convert to boolean
    lumen_bool = lumen > 0                                                                      # (B, H, W)
    plaque_bool = plaque > 0                                                                    # (B, H, W)

    # Apply logical operations
    post_processed_lumen = torch.logical_and(lumen_bool, torch.logical_not(plaque_bool))        # (B, H, W)
    post_processed_lumen = post_processed_lumen.long()                                          # (B, H, W)

    return post_processed_lumen                                                   


def load_image_as_tensor(path):
    """
        Load a grayscale image and convert to binary torch tensor.

        Args:
            path (str): The folder which contains the image.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return torch.from_numpy((img > 0).astype('uint8'))


def to_numpy(arr):
    """
        Convert PyTorch tensor to Numpy array.

        Args:
            arr (Any): The input PyTorch tensor.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.squeeze().cpu().numpy().astype(np.float32)
    return arr


def visualize_lumen_mask_plaque(lumen, mask, plaque, 
                                padding = 5, pad_value = 0.5, 
                                save_path = None):
    """
    Visualizes lumen, mask, plaque, along with the post-processed targets (lumen - plaque).

    Args:
        lumen (Tensor): (H, W) or (1, H, W)
        mask (Tensor): (H, W) or (1, H, W)
        plaque (Tensor): (H, W) or (1, H, W)
        padding (int): width of padding between images
        pad_value (float): value of padding
        save_path (int or None): the path where we save the image
    """

    lumen_np = to_numpy(lumen)
    mask_np = to_numpy(mask)
    plaque_np = to_numpy(plaque)

    lumen_tensor = lumen.unsqueeze(0).unsqueeze(0) if lumen.dim() == 2 else lumen.unsqueeze(0)
    plaque_tensor = plaque.unsqueeze(0) if plaque.dim() == 2 else plaque.unsqueeze(0)

    lumen_bool = lumen_tensor > 0
    plaque_bool = plaque_tensor > 0

    targets = post_process_targets(lumen = lumen_bool, plaque = plaque_bool)
    targets_np = to_numpy(targets)

    # Helper: horizontally stack with padding
    def add_horizontal_padding(arrays, padding, pad_value):
        padded_list = []
        for i, arr in enumerate(arrays):
            arr = arr.astype(np.float32)
            padded_list.append(arr)
            if i < len(arrays) - 1:
                pad_arr = np.ones((arr.shape[0], padding), dtype=np.float32) * pad_value
                padded_list.append(pad_arr)
        return np.hstack(padded_list)

    # Stack images horizontally
    images = [lumen_np, mask_np, plaque_np, targets_np]
    row = add_horizontal_padding(images, padding, pad_value)

    # Plot each image individually with title at center
    titles = [
        "Lumen",
        "Mask",
        "Plaque",
        "processed_targets = [Lumen AND (NOT plaque)]"
    ]

    # Create figure with gray background
    fig, ax = plt.subplots(1, 1, figsize = (21, 16))
    fig.patch.set_facecolor((pad_value, pad_value, pad_value))
    ax.imshow(row, cmap = 'gray', vmin = 0, vmax = 1)
    ax.axis('off')

    # Draw titles above each image
    H, W = row.shape
    current_x = 0
    for img, title in zip(images, titles):
        w = img.shape[1]
        ax.text(current_x + w/2, -padding * 2, 
                title, 
                ha = 'center', 
                va = 'bottom', 
                fontsize = 10, 
                color = 'black')
        current_x += w + padding

    plt.tight_layout(pad = 2)
    if save_path:
        plt.savefig(save_path, 
                    bbox_inches = 'tight', 
                    dpi = 150, 
                    facecolor = (pad_value, pad_value, pad_value))
        plt.close(fig)
    else:
        plt.show()

