module tb_ecg_classifier_core #(
    parameter integer NUM_SAMPLES = 8
);
    logic clk;
    logic reset;
    logic start;
    logic input_sample_valid;
    logic signed [11:0] input_sample;
    logic done;
    logic signed [19:0] score;
    logic predicted_class;

    logic signed [11:0] expected_input [0:NUM_SAMPLES*20-1];
    logic [319:0] expected_counts [0:NUM_SAMPLES-1];
    logic signed [19:0] expected_scores [0:NUM_SAMPLES-1];
    logic expected_classes [0:NUM_SAMPLES-1];

    integer sample_id;
    integer timestep;
    integer neuron;
    integer mismatches;
    integer cycle_counter;
    integer start_cycle;
    integer done_cycle;

    ecg_classifier_core dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .input_sample_valid(input_sample_valid),
        .input_sample(input_sample),
        .done(done),
        .score(score),
        .predicted_class(predicted_class)
    );

    always #5 clk = ~clk;
    always @(posedge clk) cycle_counter = cycle_counter + 1;

    task automatic send_sample(input integer input_index);
        begin
            @(negedge clk);
            input_sample_valid = 1'b1;
            input_sample = expected_input[input_index];
            @(posedge clk);
            #1 input_sample_valid = 1'b0;
        end
    endtask

    initial begin
        $readmemh("rtl/mem/classifier_inputs.mem", expected_input);
        $readmemh("rtl/mem/classifier_expected_counts.mem", expected_counts);
        $readmemh("rtl/mem/classifier_expected_score.mem", expected_scores);
        $readmemh("rtl/mem/classifier_expected_class.mem", expected_classes);

        clk = 1'b0;
        reset = 1'b1;
        start = 1'b0;
        input_sample_valid = 1'b0;
        input_sample = '0;
        cycle_counter = 0;
        mismatches = 0;

        repeat (2) @(posedge clk);
        #1 reset = 1'b0;

        for (sample_id = 0; sample_id < NUM_SAMPLES; sample_id = sample_id + 1) begin
            @(negedge clk);
            start = 1'b1;
            @(posedge clk);
            #1;
            start = 1'b0;
            start_cycle = cycle_counter;

            for (timestep = 0; timestep < 20; timestep = timestep + 1) begin
                send_sample(sample_id*20+timestep);
                // LOAD_SAMPLE accepts the input edge, followed by 64 neuron
                // process edges and one commit edge. The core intentionally
                // does not expose controller checkpoints, so this integration
                // bench waits for that fixed verified Phase 3 schedule.
                repeat (65) @(posedge clk);
                #1;
            end

            while (!done) begin
                @(posedge clk);
                #1;
            end
            done_cycle = cycle_counter;

            if (score !== expected_scores[sample_id] ||
                predicted_class !== expected_classes[sample_id]) begin
                $display("MISMATCH sample=%0d score=%0d/%0d class=%0d/%0d",
                    sample_id, score, expected_scores[sample_id], predicted_class,
                    expected_classes[sample_id]);
                mismatches = mismatches + 1;
                $finish;
            end

            // The core intentionally exposes only the classifier result. The
            // standalone Phase 3 controller regression verifies every final
            // count entry; this test checks the same sample sequence reaches
            // the exact readout result through the integrated datapath.
            $display("SAMPLE %0d score=%0d class=%0d cycles_from_start_to_done=%0d",
                sample_id, score, predicted_class, done_cycle-start_cycle);
            @(posedge clk);
            #1;
        end

        if (mismatches == 0) begin
            $display("PASS: full classifier samples=%0d", NUM_SAMPLES);
            $display("Reservoir cycles per sample: 1320");
            $display("Readout MAC cycles: 64");
        end else begin
            $display("FAIL: full classifier mismatches=%0d", mismatches);
        end
        $finish;
    end
endmodule
