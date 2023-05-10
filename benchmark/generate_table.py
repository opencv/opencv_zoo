import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.use("svg")

# parse a '.md' file and find a table. return table information
def parse_table(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")

    header = []
    body = []

    found_start = False  # if found table start line
    parse_done = False  # if parse table done
    for l in lines:
        if found_start and parse_done:
            break
        l = l.strip()
        if not l:
            continue
        if l.startswith("|") and l.endswith("|"):
            if not found_start:
                found_start = True
            row = [c.strip() for c in l.split("|") if c.strip()]
            if not header:
                header = row
            else:
                body.append(row)
        elif found_start:
            parse_done = True
    return header, body


# parse models information
def parse_data(models_info):
    min_list = []
    max_list = []
    colors = []
    for model in models_info:
        # remove \*
        data = [x.replace("\\*", "") for x in model]
        # get max data
        max_data = -1
        max_idx = -1
        min_data = 9999999
        min_idx = -1

        for i in range(len(data)):
            try:
                d = float(data[i])
                if d > max_data:
                    max_data = d
                    max_idx = i
                if d < min_data:
                    min_data = d
                    min_idx = i
            except:
                pass

        min_list.append(min_idx)
        max_list.append(max_idx)

        # calculate colors
        color = []
        for t in data:
            try:
                t = (float(t) - min_data) / (max_data - min_data)
                color.append(cmap(t))
            except:
                color.append('white')
        colors.append(color)
    return colors, min_list, max_list


if __name__ == '__main__':
    hardware_info, models_info = parse_table("./README.md")
    cmap = mpl.colormaps.get_cmap("RdYlGn_r")
    # remove empty line
    models_info.pop(0)
    # remove reference
    hardware_info = [re.sub(r'\[(.+?)]\(.+?\)', r'\1', r) for r in hardware_info]
    models_info = [[re.sub(r'\[(.+?)]\(.+?\)', r'\1', c) for c in r] for r in models_info]

    table_colors, min_list, max_list = parse_data(models_info)
    table_texts = [hardware_info] + models_info
    table_colors = [['white'] * len(hardware_info)] + table_colors
    # create a color bar. base width set to 1000, color map height set to 80
    fig, axs = plt.subplots(nrows=3, figsize=(10, 0.8))
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    axs[0].imshow(gradient, aspect='auto', cmap=cmap)
    axs[0].text(-0.01, 0.5, "Faster", va='center', ha='right', fontsize=11, transform=axs[0].transAxes)
    axs[0].text(1.01, 0.5, "Slower", va='center', ha='left', fontsize=11, transform=axs[0].transAxes)

    # initialize a table
    table = axs[1].table(cellText=table_texts,
                         cellColours=table_colors,
                         cellLoc="left",
                         loc="upper left")

    # adjust table position
    table_pos = axs[1].get_position()
    axs[1].set_position([
        table_pos.x0,
        table_pos.y0 - table_pos.height,
        table_pos.width,
        table_pos.height
    ])

    table.set_fontsize(11)
    table.auto_set_font_size(False)
    table.scale(1, 2)
    table.auto_set_column_width(list(range(len(table_texts[0]))))
    table.AXESPAD = 0  # cancel padding

    # highlight the best number
    for i in range(len(min_list)):
        cell = table.get_celld()[(i + 1, min_list[i])]
        cell.set_text_props(weight='bold', color='white')

    table_height = 0
    table_width = 0
    # calculate table height and width
    for i in range(len(table_texts)):
        cell = table.get_celld()[(i, 0)]
        table_height += cell.get_height()
    for i in range(len(table_texts[0])):
        cell = table.get_celld()[(0, i)]
        table_width += cell.get_width() + 0.1

    # add notes for table
    axs[2].text(0, -table_height - 0.8, "\*: Models are quantized in per-channel mode, which run slower than per-tensor quantized models on NPU.", va='bottom', ha='left', fontsize=11, transform=axs[1].transAxes)

    # turn off labels
    for ax in axs:
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])

    # adjust color map position to center
    cm_pos = axs[0].get_position()
    axs[0].set_position([
        (table_width - 1) / 2,
        cm_pos.y0,
        cm_pos.width,
        cm_pos.height
    ])

    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig("./color_table.svg", format='svg', bbox_inches="tight", pad_inches=0, metadata={'Date': None, 'Creator': None})
