import torch
from torch.autograd import Variable
from tqdm import tqdm


def train(common_net, src_net, tgt_net, optimizer, criterion, epoch,
          source_dataloader, target_dataloader, train_hist, config):
    """
    Fonction d'entraînement du modèle DAN
    Args:
        common_net: Modèle extracteur partagé
        src_net: Classifieur source
        tgt_net: Classifieur cible
        optimizer: Optimiseur
        criterion: Fonction de perte
        epoch: Numéro de l'epoch
        source_dataloader: DataLoader pour les données source
        target_dataloader: DataLoader pour les données cible
        train_hist: Historique des métriques d'entraînement
        config: Configuration (classe ou dict)
    """
    common_net.train()
    src_net.train()
    tgt_net.train()

    source_iter = iter(source_dataloader)
    target_iter = iter(target_dataloader)

    for batch_idx in tqdm(range(min(len(source_dataloader), len(target_dataloader))), 
                        desc=f'Epoch {epoch}'):
        # Récupération des données
        sdata = next(source_iter)
        tdata = next(target_iter)

        # Préparation des données
        input1, label1 = sdata
        input2, label2 = tdata
        
        if config.use_gpu:
            input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
        else:
            input1, label1 = Variable(input1), Variable(label1)
            input2, label2 = Variable(input2), Variable(label2)

        optimizer.zero_grad()

        # Forward pass
        input1 = input1.expand(input1.shape[0], 3, 28, 28)
        input = torch.cat((input1, input2), 0)
        common_feature = common_net(input)

        src_feature, tgt_feature = torch.split(common_feature, int(config.batch_size))

        src_output = src_net(src_feature)
        tgt_output = tgt_net(tgt_feature)

        # Calcul des pertes
        class_loss = criterion(src_output, label1)
        mmd_loss_value = mmd_loss(src_feature, tgt_feature) * config.theta1 + \
                       mmd_loss(src_output, tgt_output) * config.theta2

        loss = class_loss + mmd_loss_value
        loss.backward()
        optimizer.step()

        # Enregistrement des métriques
        train_hist['Total_loss'].append(loss.item())
        train_hist['Class_loss'].append(class_loss.item())
        train_hist['MMD_loss'].append(mmd_loss_value.item())

        # Affichage périodique
        if batch_idx % config.plot_iter == 0:
            print(f'[{batch_idx * len(input2)}/{len(target_dataloader.dataset)} '
                  f'({100. * batch_idx / len(target_dataloader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\t'
                  f'Class Loss: {class_loss.item():.6f}\t'
                  f'MMD Loss: {mmd_loss_value.item():.6f}')

    return common_net, src_net, tgt_net, train_hist



def test(common_net, src_net, source_dataloader, target_dataloader, epoch, test_hist, config):
    """
    Fonction d'évaluation du modèle
    """
    common_net.eval()
    src_net.eval()

    source_correct = 0
    target_correct = 0

    with torch.no_grad():
        # Évaluation sur les données source
        for batch_idx, sdata in enumerate(source_dataloader):
            input1, label1 = sdata
            if config.use_gpu:
                input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            else:
                input1, label1 = Variable(input1), Variable(label1)

            input1 = input1.expand(input1.shape[0], 3, 28, 28)
            output1 = src_net(common_net(input1))
            pred1 = output1.data.max(1, keepdim=True)[1]
            source_correct += pred1.eq(label1.data.view_as(pred1)).cpu().sum()

        # Évaluation sur les données cible
        for batch_idx, tdata in enumerate(target_dataloader):
            input2, label2 = tdata
            if config.use_gpu:
                input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
            else:
                input2, label2 = Variable(input2), Variable(label2)

            output2 = src_net(common_net(input2))
            pred2 = output2.data.max(1, keepdim=True)[1]
            target_correct += pred2.eq(label2.data.view_as(pred2)).cpu().sum()

    # Calcul des précisions
    source_accuracy = 100. * source_correct / len(source_dataloader.dataset)
    target_accuracy = 100. * target_correct / len(target_dataloader.dataset)

    print(f'\nSource Accuracy: {source_correct}/{len(source_dataloader.dataset)} '
          f'({source_accuracy:.4f}%)\n'
          f'Target Accuracy: {target_correct}/{len(target_dataloader.dataset)} '
          f'({target_accuracy:.4f}%)\n')

    test_hist['Source Accuracy'].append(source_accuracy)
    test_hist['Target Accuracy'].append(target_accuracy)

    return test_hist