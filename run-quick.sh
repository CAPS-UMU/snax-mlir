if [[ $1 == "-z" ]]; then
    # set up
    # INPUT="/repo/kernels/$2/$2.mlir"
    # OUT="/repo/kernels/$2/out-my-lowering"
    # mkdir -p $OUT
    # rm $OUT/*
    # # remove old logs    
    # rm -fr /repo/kernels/$2/*.logs/
    
    # # run input through my lowering passes
    # mlir-opt $INPUT --mlir-print-op-generic > "$OUT/$2_zigzag2.mlir"
    # mlir-opt "$OUT/$2_zigzag2.mlir" --linalg-generalize-named-ops --mlir-print-op-generic --mlir-print-local-scope >"$OUT/$2_zigzag3.mlir"
    # # mlir-opt "$OUT/$2_zigzag3.mlir" --one-shot-bufferize='bufferize-function-boundaries' > "$OUT/$2_zigzag4.mlir"
    # # save processed input as next input to built in snax-mlir flow
    # cat "$OUT/$2_zigzag3.mlir" > "/repo/kernels/$2/$2_zigzag.mlir"
    # # build and run example
    # cd /repo/kernels/$2
    # sh /repo/snax-gemm-step-by-step.sh "$2_zigzag" "/repo/kernels/$2"
    # cd /repo
    echo "in -z mode"
    
else
    if [[ $1 == "streamer_matmul" ]]; then
    ## COMPILE AND RUN REGULAR MATMUL
    # create output folders if they do not already exist
    OUT="/repo/kernels/$1/OUT"
    OUT2="/repo/kernels/$1/OUT-TILED"
    mkdir -p $OUT
    mkdir -p $OUT2
    # empty output folder
    rm $OUT/*
    rm $OUT2/*
    # remove any old log files
    rm -fr /repo/kernels/$1/*.logs/
    # navigate to kernel directory
    cd "/repo/kernels/$1"
    # create data.h and data.c
    gendata.py
    # lower matmul mlir to llvm, the compile.
    mlir-opt --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named, tosa-to-tensor, tosa-to-scf, tosa-to-linalg))' --mlir-print-op-generic --mlir-print-local-scope -o $OUT/matmul.preproc1.mlir matmul.mlir
    mlir-opt --tosa-to-arith="include-apply-rescale"  --empty-tensor-to-alloc-tensor -o $OUT/matmul.preproc2.mlir $OUT/matmul.preproc1.mlir
    mlir-opt --test-linalg-transform-patterns="test-generalize-pad-tensor" --linalg-generalize-named-ops --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops function-boundary-type-conversion=identity-layout-map" --mlir-print-op-generic --mlir-print-local-scope -o $OUT/matmul.preprocfinal.mlir $OUT/matmul.preproc2.mlir
    /repo/runtime//../compiler/snax-opt -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=3},convert-linalg-to-stream,convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space -o $OUT/matmul.snax-opt.mlir $OUT/matmul.preprocfinal.mlir
    mlir-opt --convert-linalg-to-loops --convert-scf-to-cf --lower-affine --canonicalize --cse --convert-math-to-llvm --llvm-request-c-wrappers --expand-strided-metadata --lower-affine --convert-index-to-llvm=index-bitwidth=32 --convert-cf-to-llvm=index-bitwidth=32 --convert-arith-to-llvm=index-bitwidth=32 --convert-func-to-llvm='index-bitwidth=32' --finalize-memref-to-llvm='use-generic-functions index-bitwidth=32' --canonicalize --reconcile-unrealized-casts -o $OUT/matmul.ll.mlir $OUT/matmul.snax-opt.mlir
    mlir-translate --mlir-to-llvmir -o $OUT/matmul.ll $OUT/matmul.ll.mlir
    /repo/runtime//tollvm12.py < $OUT/matmul.ll > $OUT/matmul.ll12
    /usr/bin/clang -Wno-unused-command-line-argument -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/common -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/riscv64/bits/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/generic -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/include -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/internal -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include/bits -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-common -O3 -std=gnu11 -Wall -Wextra -x ir -c $OUT/matmul.ll12 -o $OUT/matmul.o
    /usr/bin/clang -Wno-unused-command-line-argument -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/common -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/riscv64/bits/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/generic -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/include -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/internal -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include/bits -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-common -O3 -std=gnu11 -Wall -Wextra -c main.c -o $OUT/main.o
    /usr/bin/clang -Wno-unused-command-line-argument -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/common -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/riscv64/bits/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/generic -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/include -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/internal -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include/bits -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-common -O3 -std=gnu11 -Wall -Wextra -c data.c -o $OUT/data.o
    /usr/bin/clang -fuse-ld=/usr/bin/ld.lld --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -T/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/base.ld -L/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/rtl-generic -L/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/rtl-generic/build -nostdlib -lsnRuntime $OUT/matmul.o $OUT/main.o $OUT/data.o -o matmul.x
    # run the matmul executable
    /opt/snax-kul-cluster-mixed-narrow-wide-rtl/bin/snitch_cluster.vlt matmul.x
    
    ## COMPILE AND RUN TILED MATMUL
    mlir-opt --pass-pipeline='builtin.module(test-transform-dialect-interpreter{bind-first-extra-to-ops=linalg.quantized_matmul}, test-transform-dialect-erase-schedule)' -o $OUT2/transform_matmul.mlir transform_matmul.transform.mlir
    mlir-opt --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named, tosa-to-tensor, tosa-to-scf, tosa-to-linalg))' --mlir-print-op-generic --mlir-print-local-scope -o $OUT2/transform_matmul.preproc1.mlir $OUT2/transform_matmul.mlir
    mlir-opt --tosa-to-arith="include-apply-rescale"  --empty-tensor-to-alloc-tensor -o $OUT2/transform_matmul.preproc2.mlir $OUT2/transform_matmul.preproc1.mlir
    mlir-opt --test-linalg-transform-patterns="test-generalize-pad-tensor" --linalg-generalize-named-ops --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops function-boundary-type-conversion=identity-layout-map" --mlir-print-op-generic --mlir-print-local-scope -o $OUT2/transform_matmul.preprocfinal.mlir $OUT2/transform_matmul.preproc2.mlir
    /repo/runtime//../compiler/snax-opt -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=3},convert-linalg-to-stream,convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space -o $OUT2/transform_matmul.snax-opt.mlir $OUT2/transform_matmul.preprocfinal.mlir
    mlir-opt --convert-linalg-to-loops --convert-scf-to-cf --lower-affine --canonicalize --cse --convert-math-to-llvm --llvm-request-c-wrappers --expand-strided-metadata --lower-affine --convert-index-to-llvm=index-bitwidth=32 --convert-cf-to-llvm=index-bitwidth=32 --convert-arith-to-llvm=index-bitwidth=32 --convert-func-to-llvm='index-bitwidth=32' --finalize-memref-to-llvm='use-generic-functions index-bitwidth=32' --canonicalize --reconcile-unrealized-casts -o $OUT2/transform_matmul.ll.mlir $OUT2/transform_matmul.snax-opt.mlir
    mlir-translate --mlir-to-llvmir -o $OUT2/transform_matmul.ll $OUT2/transform_matmul.ll.mlir
    /repo/runtime//tollvm12.py < $OUT2/transform_matmul.ll > $OUT2/transform_matmul.ll12
    /usr/bin/clang -Wno-unused-command-line-argument -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/common -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/riscv64/bits/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/generic -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/include -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/internal -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include/bits -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-common -O3 -std=gnu11 -Wall -Wextra -x ir -c $OUT2/transform_matmul.ll12 -o $OUT2/transform_matmul.o
    /usr/bin/clang -fuse-ld=/usr/bin/ld.lld --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -T/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/base.ld -L/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/rtl-generic -L/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/rtl-generic/build -nostdlib -lsnRuntime $OUT2/transform_matmul.o $OUT/main.o $OUT/data.o -o transform_matmul.x
    rm -fr ./logs/
    /opt/snax-kul-cluster-mixed-narrow-wide-rtl/bin/snitch_cluster.vlt transform_matmul.x
    echo "$1 !!! was passed in"
    cd /repo
    
    else
    cd /repo/kernels/$1
    rm -fr ./*.logs/
    make allrun
    cd /repo
    fi
fi

# mlir-opt --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named, tosa-to-tensor, tosa-to-scf, tosa-to-linalg))' --mlir-print-op-generic --mlir-print-local-scope -o matmul.preproc1.mlir matmul.mlir
# mlir-opt --tosa-to-arith="include-apply-rescale"  --empty-tensor-to-alloc-tensor -o matmul.preproc2.mlir matmul.preproc1.mlir
# mlir-opt --test-linalg-transform-patterns="test-generalize-pad-tensor" --linalg-generalize-named-ops --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops function-boundary-type-conversion=identity-layout-map" --mlir-print-op-generic --mlir-print-local-scope -o matmul.preprocfinal.mlir matmul.preproc2.mlir
# /repo/runtime//../compiler/snax-opt -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=3},convert-linalg-to-stream,convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space -o matmul.snax-opt.mlir matmul.preprocfinal.mlir
# mlir-opt --convert-linalg-to-loops --convert-scf-to-cf --lower-affine --canonicalize --cse --convert-math-to-llvm --llvm-request-c-wrappers --expand-strided-metadata --lower-affine --convert-index-to-llvm=index-bitwidth=32 --convert-cf-to-llvm=index-bitwidth=32 --convert-arith-to-llvm=index-bitwidth=32 --convert-func-to-llvm='index-bitwidth=32' --finalize-memref-to-llvm='use-generic-functions index-bitwidth=32' --canonicalize --reconcile-unrealized-casts -o matmul.ll.mlir matmul.snax-opt.mlir
# mlir-translate --mlir-to-llvmir -o matmul.ll matmul.ll.mlir
# /repo/runtime//tollvm12.py < matmul.ll > matmul.ll12
# /usr/bin/clang -Wno-unused-command-line-argument -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/common -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/riscv64/bits/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/generic -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/include -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/internal -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include/bits -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-common -O3 -std=gnu11 -Wall -Wextra -x ir -c matmul.ll12 -o matmul.o
# /usr/bin/clang -Wno-unused-command-line-argument -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-kul-cluster-mixed-narrow-wide//target/snitch_cluster/sw/runtime/common -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/src/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/snRuntime/api/omp/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/riscv64/bits/ -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/arch/generic -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/include -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/src/internal -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include/bits -I/opt/snax-kul-cluster-mixed-narrow-wide//sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-common -O3 -std=gnu11 -Wall -Wextra -c main.c -o main.o

