import matplotlib.pyplot as plt
import numpy as np
import torch
import copy

def get_pde_points(pde, n_points):
    inputs = pde.soboleng.draw(n_points)
    inputs = pde.convert(inputs)
    output = pde.approximate_solution(inputs).reshape(-1, )
    exact_output = pde.exact_solution(inputs).reshape(-1, )
    return inputs, output, exact_output


def get_coefficient_points(pde, n_points):
    inputs = pde.soboleng.draw(n_points)
    inputs = pde.convert(inputs)
    output = pde.approximate_coefficient(inputs).reshape(-1, )
    exact_output = pde.exact_coefficient(inputs).reshape(-1, )
    return inputs, output, exact_output


def relative_L2_error(pde, n_points=10000, function='pde'):
    if function == 'pde':
        inputs, output, exact_output = get_pde_points(pde, n_points)
    else:
        inputs, output, exact_output = get_coefficient_points(pde, n_points)
    err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5
    print(f'{function} loss:')
    print('L2 Relative Error Norm: {:.6e}'.format(err.item()))
    with open(f'{pde.path}/relative_error.txt', 'a') as f:
        f.write(f'{function} loss:\n')
        f.write('L2 Relative Error Norm: {:.6e}\n'.format(err.item()))
    return inputs, output, exact_output


def plotting_solution(pde, n_points=50000, function='pde'):
    inputs, output, exact_output = relative_L2_error(pde, n_points, function)
    inputs = inputs.cpu()
    output = output.cpu()
    exact_output = exact_output.cpu()
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=exact_output.detach(), cmap='jet')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('t')
    plt.colorbar(im1, ax=axs[0])
    axs[0].grid(True, which='both', ls=':')
    im2 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output.detach(), cmap='jet')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('t')
    plt.colorbar(im2, ax=axs[1])
    axs[1].grid(True, which='both', ls=':')
    title = 'Solution' if function == 'pde' else 'Coefficient'
    axs[0].set_title(f'Exact {title}')
    axs[1].set_title(f'Approximate {title}')

    plt.savefig(f'{pde.path}/{title}.png')
    plt.show()
    plt.close()


def plot_training_points(pde):
    # Plot the input training points
    input_sb_left, _ = pde.add_spatial_boundary_points_left()
    input_sb_right_, _ = pde.add_spatial_boundary_points_right()
    input_tb_, _ = pde.add_temporal_boundary_points()
    input_int_, _ = pde.add_interior_points()
    input_meas_, output_meas_ = pde.get_measurement_data()

    input_sb_left_ = copy.deepcopy(input_sb_left).cpu()
    input_sb_right_ = copy.deepcopy(input_sb_right_).cpu()
    input_tb_ = copy.deepcopy(input_tb_).cpu()
    input_int_ = copy.deepcopy(input_int_).cpu()
    input_meas_ = copy.deepcopy(input_meas_).cpu()

    plt.figure(figsize=(16, 8), dpi=150)
    plt.scatter(input_sb_left_[:, 1].detach().numpy(), input_sb_left_[:, 0].detach().numpy(),
                label='Left Boundary Points')
    plt.scatter(input_sb_right_[:, 1].detach().numpy(), input_sb_right_[:, 0].detach().numpy(),
                label='Right Boundary Points')
    plt.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 0].detach().numpy(), label='Interior Points')
    plt.scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 0].detach().numpy(), label='Initial Points')
    plt.scatter(input_meas_[:, 1].detach().numpy(), input_meas_[:, 0].detach().numpy(), label='Measurement Points')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend()
    plt.savefig(f'{pde.path}/training_points.png')
    plt.show()


def plot_train_loss(pde, hist):
    hist = hist['total_loss']
    plt.figure(dpi=150)
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel('Log10 Loss')
    plt.legend()
    plt.savefig(f'{pde.path}/train_loss.png')
    plt.show()
