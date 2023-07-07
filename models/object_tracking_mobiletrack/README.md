# MobileTrack

Two versions of the tracker ONNX model are currently available, mobiletrack and mobiletrack_fast, and their performance is shown in the table below.

| Model            | TrackingNet | GOT-10k | LaSOT | MACs | Params | 1 Thread Speed | 2 Threads Speed | 4 Threads Speed |
| ---------------- | ----------- | ------- | ----- | ---- | ------ | -------------- | --------------- | --------------- |
| mobiletrack      | 65.6        | 54.2    | 48.8  | 90M  | 364K   | 4.95 ms        | 3.12 ms         | 2.32 ms         |
| mobiletrack_fast | 65.3        | 52.0    | 46.0  | 67M  | 261K   | 3.65 ms        | 2.51 ms         | 1.91 ms         |

The speed of the model in table is measured on the Intel® Core™ i7-9750H CPU @ 2.60GHz × 12 .