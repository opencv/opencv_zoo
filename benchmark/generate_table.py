import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import yaml


# parse a '.md' file and find a table. return table information
def parse_table(filepath, cfg):
    #  parse benchmark data
    def _parse_benchmark_data(lines):
        raw_data = []
        for l in lines:
            l = l.strip()
            # parse each line
            m = re.match(r"(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+\[([^]]*)]\s+(.*)", l)
            if m:
                raw_data.append(m.groups())
        return raw_data

    # find each cpu, gpu, npu block
    def _find_all_platform_block(lines):
        cur_start = None
        cur_platform = None
        platform_block = dict()
        for i in range(len(lines)):
            l = lines[i].strip()
            # found start and end of a platform
            if l.startswith("CPU") or l.startswith("GPU") or l.startswith("NPU"):
                if cur_platform is not None:
                    platform_block[cur_platform] = (cur_start, i)
                cur_platform = l[:-1]
                cur_start = i + 1
                continue
            if cur_platform is not None and i == len(lines) - 1:
                platform_block[cur_platform] = (cur_start, i)
        for key in platform_block:
            r = platform_block[key]
            platform_block[key] = _parse_benchmark_data(lines[r[0]:r[1]])

        return platform_block

    # find device block
    def _find_all_device_block(lines, level):
        cur_start = None
        cur_device_name = None
        device_block = dict()
        for i in range(len(lines)):
            l = lines[i].strip()
            m = re.match(r"^(#+)\s+(.*)", l)
            # found start and end of a device
            if m and len(m.group(1)) == level:
                if cur_device_name is not None:
                    device_block[cur_device_name] = (cur_start, i)
                cur_device_name = m.group(2)
                cur_start = i + 1
                continue
            if cur_device_name is not None and i == len(lines) - 1:
                device_block[cur_device_name] = (cur_start, i)

        for key in device_block:
            r = device_block[key]
            device_block[key] = _find_all_platform_block(lines[r[0]:r[1]])

        return device_block

    # find detail block
    def _find_detail_block(lines, title, level):
        start = None
        end = len(lines)
        for i in range(len(lines)):
            l = lines[i].strip()
            m = re.match(r"^(#+)\s+(.*)", l)
            # found start of detailed results block
            if m and len(m.group(1)) == level and m.group(2) == title:
                start = i + 1
                continue
            # found end of detailed results block
            if start is not None and m and len(m.group(1)) <= level:
                end = i
                break

        return _find_all_device_block(lines[start:end], level + 1)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")

    devices = cfg["Devices"]
    models = cfg["Models"]
    # display information of all devices
    devices_display = [x['display_info'] for x in cfg["Devices"]]
    header = ["Model", "Task", "Input Size"] + devices_display
    body = [[x["name"], x["task"], x["input_size"]] + ["---"] * len(devices) for x in models]
    table_raw_data = _find_detail_block(lines, title="Detailed Results", level=2)

    device_name_header = [f"{x['name']}-{x['platform']}" for x in devices]
    device_name_header = [""] * (len(header) - len(device_name_header)) + device_name_header
    # device name map to model col idx
    device_name_to_col_idx = {k: v for v, k in enumerate(device_name_header)}
    # model name map to model row idx
    model_name_to_row_idx = {k[0]: v for v, k in enumerate(body)}
    # convert raw data to usage data
    for device in devices:
        raw_data = table_raw_data[device["name"]][device["platform"]]
        col_idx = device_name_to_col_idx[f"{device['name']}-{device['platform']}"]
        for model in models:
            # find which row idx of this model
            row_idx = model_name_to_row_idx[model["name"]]
            model_idxs = [i for i in range(len(raw_data)) if model["keyword"] in raw_data[i][-1]]
            if len(model_idxs) > 0:
                # only choose the first one
                model_idx = model_idxs[0]
                # choose mean as value
                body[row_idx][col_idx] = raw_data[model_idx][0]
                # remove used data
            for idx in sorted(model_idxs, reverse=True):
                raw_data.pop(idx)

    # handle suffix
    for suffix in cfg["Suffixes"]:
        row_idx = model_name_to_row_idx[suffix["model"]]
        col_idx = device_name_to_col_idx[f"{suffix['device']}-{suffix['platform']}"]
        body[row_idx][col_idx] += suffix["str"]

    return header, body


# render table and save
def render_table(header, body, save_path, cfg, cmap_type):
    # parse models information and return some data
    def _parse_data(models_info, cmap, cfg):
        min_list = []
        max_list = []
        colors = []
        # model name map to idx
        model_name_to_idx = {k["name"]: v for v, k in enumerate(cfg["Models"])}
        for model in models_info:
            # remove \*
            data = [x.replace("\\*", "") for x in model]
            # get max data
            max_idx = -1
            min_data = 9999999
            min_idx = -1

            for i in range(len(data)):
                try:
                    d = float(data[i])
                    if d < min_data:
                        min_data = d
                        min_idx = i
                except:
                    pass
            # set all bigger than acceptable time to red color
            idx = model_name_to_idx[model[0]]
            acc_time = cfg["Models"][idx]["acceptable_time"]

            min_list.append(min_idx)
            max_list.append(max_idx)

            # calculate colors
            color = []
            for t in data:
                try:
                    t = float(t)
                    if t > acc_time:
                        # all bigger time will be set to red
                        color.append(cmap(1.))
                    else:
                        # sqrt to make the result non-linear
                        t = np.sqrt((t - min_data) / (acc_time - min_data))
                        color.append(cmap(t))
                except:
                    color.append('white')
            colors.append(color)
        return colors, min_list, max_list

    cmap = mpl.colormaps.get_cmap(cmap_type)
    table_colors, min_list, max_list = _parse_data(body, cmap, cfg)
    table_texts = [header] + body
    table_colors = [['white'] * len(header)] + table_colors

    # create a figure, base width set to 1000, height set to 80
    fig, axs = plt.subplots(nrows=3, figsize=(10, 0.8))
    # turn off labels and axis
    for ax in axs:
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])

    # create and add a color map
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
    # set style of header, each url of hardware
    ori_height = table[0, 0].get_height()
    url_base = 'https://github.com/opencv/opencv_zoo/tree/main/benchmark#'
    hw_urls = [f"{url_base}{x['name'].lower().replace(' ', '-')}" for x in cfg["Devices"]]
    hw_urls = [""] * 3 + hw_urls
    for col in range(len(header)):
        cell = table[0, col]
        cell.set_text_props(ha='center', weight='bold', linespacing=1.5, url=hw_urls[col])
        cell.set_url(hw_urls[col])
        cell.set_height(ori_height * 2.2)

    url_base = 'https://github.com/opencv/opencv_zoo/tree/main/models/'
    model_urls = [f"{url_base}{x['folder']}" for x in cfg["Models"]]
    model_urls = [""] + model_urls
    for row in range(len(body) + 1):
        cell = table[row, 0]
        cell.set_text_props(url=model_urls[row])
        cell.set_url(model_urls[row])

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

    # draw table and trigger changing the column width value
    fig.canvas.draw()
    # calculate table height and width
    table_height = 0
    table_width = 0
    for i in range(len(table_texts)):
        cell = table.get_celld()[(i, 0)]
        table_height += cell.get_height()
    for i in range(len(table_texts[0])):
        cell = table.get_celld()[(0, i)]
        table_width += cell.get_width()

    # add notes for table
    axs[2].text(0, -table_height - 1, "Units: All data in milliseconds (ms).", va='bottom', ha='left', fontsize=11, transform=axs[1].transAxes)
    axs[2].text(0, -table_height - 2, "\\*: Models are quantized in per-channel mode, which run slower than per-tensor quantized models on NPU.", va='bottom', ha='left', fontsize=11, transform=axs[1].transAxes)

    # adjust color map position to center
    cm_pos = axs[0].get_position()
    axs[0].set_position([
        (table_width - 1) / 2,
        cm_pos.y0,
        cm_pos.width,
        cm_pos.height
    ])

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['svg.hashsalt'] = '11'  # fix hash salt for avoiding id change
    plt.savefig(save_path, format='svg', bbox_inches="tight", pad_inches=0, metadata={'Date': None, 'Creator': None})


if __name__ == '__main__':
    with open("table_config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    hw_info, model_info = parse_table("README.md", cfg)
    render_table(hw_info, model_info, "color_table.svg", cfg, "RdYlGn_r")
