module tb_readout_mac #(
    parameter integer NUM_VECTORS = 14
);
    logic clk;
    logic reset;
    logic start;
    logic [64*5-1:0] spike_counts;
    logic done;
    logic signed [19:0] score;
    logic predicted_class;
    logic mac_valid;
    logic [6:0] mac_index;
    logic [4:0] mac_count;
    logic signed [7:0] mac_weight;
    logic signed [31:0] mac_product;
    logic signed [19:0] mac_accumulator_before;
    logic signed [19:0] mac_accumulator_after;

    logic [319:0] expected_counts [0:NUM_VECTORS-1];
    logic signed [7:0] expected_weights [0:NUM_VECTORS*64-1];
    logic signed [31:0] expected_products [0:NUM_VECTORS*64-1];
    logic signed [19:0] expected_before [0:NUM_VECTORS*64-1];
    logic signed [19:0] expected_after [0:NUM_VECTORS*64-1];
    logic signed [19:0] expected_scores [0:NUM_VECTORS-1];
    logic expected_classes [0:NUM_VECTORS-1];

    integer vector_index;
    integer neuron;
    integer mismatches;
    integer mac_comparisons;

    readout_mac dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .spike_counts(spike_counts),
        .done(done),
        .score(score),
        .predicted_class(predicted_class),
        .mac_valid(mac_valid),
        .mac_index(mac_index),
        .mac_count(mac_count),
        .mac_weight(mac_weight),
        .mac_product(mac_product),
        .mac_accumulator_before(mac_accumulator_before),
        .mac_accumulator_after(mac_accumulator_after)
    );

    always #5 clk = ~clk;

    task automatic fail_mac(input integer sample_id, input integer n);
        begin
            $display("MISMATCH vector=%0d neuron=%0d index=%0d count=%0d/%0d weight=%0d/%0d product=%0d/%0d acc_before=%0d/%0d acc_after=%0d/%0d valid=%0d",
                sample_id, n, mac_index, mac_count, expected_counts[sample_id][n*5 +: 5],
                mac_weight, expected_weights[sample_id*64+n],
                mac_product, expected_products[sample_id*64+n],
                mac_accumulator_before, expected_before[sample_id*64+n],
                mac_accumulator_after, expected_after[sample_id*64+n], mac_valid);
            mismatches = mismatches + 1;
            $finish;
        end
    endtask

    initial begin
        $readmemh("rtl/mem/readout_counts.mem", expected_counts);
        $readmemh("rtl/mem/readout_weights.mem", expected_weights);
        $readmemh("rtl/mem/readout_expected_product.mem", expected_products);
        $readmemh("rtl/mem/readout_expected_acc_before.mem", expected_before);
        $readmemh("rtl/mem/readout_expected_acc_after.mem", expected_after);
        $readmemh("rtl/mem/readout_expected_score.mem", expected_scores);
        $readmemh("rtl/mem/readout_expected_class.mem", expected_classes);

        clk = 1'b0;
        reset = 1'b1;
        start = 1'b0;
        spike_counts = '0;
        mismatches = 0;
        mac_comparisons = 0;

        repeat (2) @(posedge clk);
        #1 reset = 1'b0;

        for (vector_index = 0; vector_index < NUM_VECTORS; vector_index = vector_index + 1) begin
            @(negedge clk);
            spike_counts = expected_counts[vector_index];
            start = 1'b1;
            @(posedge clk);
            #1 start = 1'b0;

            for (neuron = 0; neuron < 64; neuron = neuron + 1) begin
                @(posedge clk);
                #1;
                if (!mac_valid || mac_index !== neuron[6:0] ||
                    mac_count !== expected_counts[vector_index][neuron*5 +: 5] ||
                    mac_weight !== expected_weights[vector_index*64+neuron] ||
                    mac_product !== expected_products[vector_index*64+neuron] ||
                    mac_accumulator_before !== expected_before[vector_index*64+neuron] ||
                    mac_accumulator_after !== expected_after[vector_index*64+neuron])
                    fail_mac(vector_index, neuron);
                mac_comparisons = mac_comparisons + 1;
            end

            if (!done || score !== expected_scores[vector_index] ||
                predicted_class !== expected_classes[vector_index]) begin
                $display("MISMATCH final vector=%0d score=%0d/%0d class=%0d/%0d done=%0d",
                    vector_index, score, expected_scores[vector_index], predicted_class,
                    expected_classes[vector_index], done);
                mismatches = mismatches + 1;
                $finish;
            end
            @(posedge clk);
            #1;
        end

        if (mismatches == 0) begin
            $display("PASS: readout vectors=%0d MAC comparisons=%0d", NUM_VECTORS, mac_comparisons);
            $display("Zero-score rule: class 1 verified by directed vectors");
            $display("Readout saturation mismatches: 0");
        end else begin
            $display("FAIL: readout mismatches=%0d", mismatches);
        end
        $finish;
    end
endmodule
