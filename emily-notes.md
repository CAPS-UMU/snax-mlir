# notes on tiling in snax-mlir

## 0. repo setup

Clone repo: `git clone git@github.com:CAPS-UMU/snax-mlir.git`

Make your own branch: `git checkout -b "learn"`

Create a conda environment to keep python related installs in order:

```
conda create -n snax-env
```

*Note for Emily: `environment location: /home/hoppip/miniforge3/envs/snax-env`*

Use conda environment: Maybe don't need conda at all?

```
conda activate snax-env
```

when you are done, 

```
conda deactivate
```

## 1. everyday set up commands

Slow:

1. `sudo chmod 666 /var/run/docker.sock`

   ```
   DOCKER_VERSION=$(cat .github/workflows/build-run-kernel.yml | grep -o "snax:v[0-9]\(.[0-9]\)\+") && 
   docker run -itv `pwd`:/repo:z ghcr.io/kuleuven-micas/$DOCKER_VERSION
   ```

2. ```
   pip install -e /repo
   cd /repo
   pip install -r requirements.txt
   ```

Fast(er):

1. ```
   DOCKER_VERSION=$(cat .github/workflows/build-run-kernel.yml | grep -o "snax:v[0-9]\(.[0-9]\)\+") && 
   docker run -itv `pwd`:/repo:z ghcr.io/kuleuven-micas/$DOCKER_VERSION
   ```

2. ```
   cd repo; bash quick-setup.sh
   ```

## 2. Run an Example

```
bash run-quick.sh streamer_matmul
```

## 3. Examine the untiled and tiled IR

untiled:

```
func.func @streamer_matmul(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16], offset:0>>, %arg2: memref<16x16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<16x16xi8>, memref<16x16xi8, strided<[1, 16], offset:0>>, i32, i32) outs(%arg2 : memref<16x16xi32>)
    return
 }
```

tiled:

```
module {
  func.func @streamer_matmul(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16]>>, %arg2: memref<16x16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    scf.for %arg3 = %c0 to %c16 step %c8 {
      %c0_0 = arith.constant 0 : index
      %c16_1 = arith.constant 16 : index
      %c8_2 = arith.constant 8 : index
      scf.for %arg4 = %c0_0 to %c16_1 step %c8_2 {
        %c0_3 = arith.constant 0 : index
        %c16_4 = arith.constant 16 : index
        %c0_5 = arith.constant 0 : index
        %subview = memref.subview %arg0[%arg3, 0] [8, 16] [1, 1] : memref<16x16xi8> to memref<8x16xi8, strided<[16, 1], offset: ?>>
        %subview_6 = memref.subview %arg1[0, %arg4] [16, 8] [1, 1] : memref<16x16xi8, strided<[1, 16]>> to memref<16x8xi8, strided<[1, 16], offset: ?>>
        %subview_7 = memref.subview %arg2[%arg3, %arg4] [8, 8] [1, 1] : memref<16x16xi32> to memref<8x8xi32, strided<[16, 1], offset: ?>>
        linalg.quantized_matmul ins(%subview, %subview_6, %c0_i32, %c0_i32 : memref<8x16xi8, strided<[16, 1], offset: ?>>, memref<16x8xi8, strided<[1, 16], offset: ?>>, i32, i32) outs(%subview_7 : memref<8x8xi32, strided<[16, 1], offset: ?>>)
      }
    }
    return
  }
}
```

