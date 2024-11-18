import random

def introduce_text_noise(ground_truth, noise_ratio):
    """
    将相邻两条text作为一组，随机选择X%的组进行内部翻转。
    - ground_truth: ground truth的原始数据列表，格式为 [(id, text)]。
    - noise_ratio: 噪声比例（0到1之间的浮点数，例如0.2表示20%标签噪声）。
    """
    # 分组数据
    num_samples = len(ground_truth)
    paired_ground_truth = [(ground_truth[i], ground_truth[i + 1]) 
                           for i in range(0, num_samples - 1, 2)]
    
    # 计算需要翻转的组数
    num_pairs = len(paired_ground_truth)
    num_noisy_pairs = int(num_pairs * noise_ratio)
    
    # 随机选择需要翻转的组索引
    noisy_indices = random.sample(range(num_pairs), num_noisy_pairs)
    
    # 对指定的组进行内部翻转
    noisy_ground_truth = []
    for i, (item1, item2) in enumerate(paired_ground_truth):
        if i in noisy_indices:
            # 翻转组内的文本内容
            noisy_ground_truth.append((item1[0], item2[1]))
            noisy_ground_truth.append((item2[0], item1[1]))
        else:
            noisy_ground_truth.append(item1)
            noisy_ground_truth.append(item2)
    
    # 如果样本数量为奇数，直接将最后一个未分组的元素添加
    if num_samples % 2 != 0:
        noisy_ground_truth.append(ground_truth[-1])
    
    # 将结果格式化为ID TEXT格式
    formatted_noisy_ground_truth = [f"{id} {text}" for id, text in noisy_ground_truth]
    return formatted_noisy_ground_truth

# 加载原始ground truth文件
with open("./data/train_95/text", 'r', encoding='utf-8') as file:
    ground_truth = [line.strip().split(" ", 1) for line in file]

# 生成不同噪声比例的ground truth
ground_truth_noisy_20 = introduce_text_noise(ground_truth, 0.2)
ground_truth_noisy_50 = introduce_text_noise(ground_truth, 0.5)
ground_truth_noisy_80 = introduce_text_noise(ground_truth, 0.8)

# 保存新的ground truth文件
def save_ground_truth(noisy_ground_truth, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("\n".join(noisy_ground_truth) + "\n")

# 保存带有20%、50%、80%标签噪声的ground truth文件
save_ground_truth(ground_truth_noisy_20, "./data/train_95/text_noisy_20")
save_ground_truth(ground_truth_noisy_50, "./data/train_95/text_noisy_50")
save_ground_truth(ground_truth_noisy_80, "./data/train_95/text_noisy_80")