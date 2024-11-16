# 1. Tải & Tiền Xử Lý Dữ Liệu
# 2. Định Nghĩa Mô Hình U2-Net 
# 3. Biên Dịch Mô Hình
# 4. Huấn Luyện, Đánh Giá Mô hình 

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from u2net import  U2NET
from u2net_function import SegmentationDataset, save_images, calculate_bce_loss, calculate_iou_and_save


if __name__ == "__main__":
    # 1. Tải & Tiền Xử Lý Dữ Liệu
    # define paths
    train_data_path = 'dataset/train_data'
    train_mask_path = 'dataset/train_mask'
    test_data_path = 'dataset/test_data'
    test_mask_path = 'dataset/test_mask'
    # Hyperparameters:
    epochs = 10
    save_epoch_interval = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data loaders
    train_loader = DataLoader(SegmentationDataset(train_data_path, train_mask_path), shuffle=True, num_workers=4, batch_size=4)
    test_loader = DataLoader(SegmentationDataset(test_data_path, test_mask_path), shuffle=True, num_workers=4, batch_size=4)

    # 2. Định Nghĩa Mô Hình U2-Net 
    model = U2NET().to(device=device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 3. Biên Dịch Mô Hình

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = None

    # Create directory to save model
    os.makedirs('models', exist_ok=True)

    log_interval = 1

    # 4. Huấn Luyện, Đánh Giá Mô hình 
    print('-------- Starting Training --------')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for idx, (images, targets, _) in enumerate(train_loader):
            images, targets =images.to(device), targets.to(device)

            # Forward to model
            d_output = model(images)
            loss0, loss, *_ = calculate_bce_loss(d_output, targets)

            # Backpropagation 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            
            # Print iteration loss
            if (idx + 1) % log_interval == 0:  # Log every 'log_interval' iterations
                print(f'[Epoch {epoch+1}/{epochs}, Iteration {idx+1}/{len(train_loader)}] '
                    f'Batch Loss: {loss.item():.6f}, Total Loss: {total_loss / ((idx + 1) * images.size(0)):.6f}')

            # Learning rate sheduler step:
            if scheduler:
                scheduler.step(total_loss /len(train_loader.dataset))
        print(f'[Epoch {epoch+1}/{epochs}] Average loss: {total_loss /len(train_loader): .6f}')

        #save model
        if (epoch + 1) % save_epoch_interval == 0:
            model_save_path = f'models/u2net_{epoch+1}.pth'
            torch.save(model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(), model_save_path)

            results_folder = os.path.join(('results', f'testing_result_{epoch+1}'))
            os.makedirs(results_folder, exist_ok=True)

            # Evaluation model 
            model.eval()

            #Eveluate and save test results

            with torch.no_grad():
                for images, targets, paths in test_loader:
                    images, targets =images.to(device), targets.to(device)
                    d_output = model(images)
                    calculate_iou_and_save(d_output[0], targets, paths, results_folder)
                    save_images(d_output[0], paths, results_folder)
    

    
