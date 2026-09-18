module tb_reservoir_controller #(
    parameter integer NUM_SEGMENTS = 14,
    parameter integer NUM_CHECKPOINTS = NUM_SEGMENTS * 20
);
    logic clk;
    logic reset;
    logic start;
    logic input_sample_valid;
    logic signed [11:0] input_sample;
    logic done;
    logic [64*5-1:0] spike_counts;
    logic checkpoint_valid;
    logic [4:0] checkpoint_timestep;
    logic [2:0] debug_state;
    logic [5:0] debug_neuron_counter;
    logic [4:0] debug_timestep_counter;
    logic [64*16-1:0] checkpoint_membrane_before;
    logic [64*16-1:0] checkpoint_membrane_after;
    logic [64*16-1:0] checkpoint_membrane;
    logic [64*32-1:0] checkpoint_input;
    logic [64*32-1:0] checkpoint_recurrent;
    logic [63:0] checkpoint_spikes;
    logic [64*5-1:0] checkpoint_spike_counts;
    logic [63:0] checkpoint_previous_spikes;

    logic signed [11:0] expected_input [0:NUM_SEGMENTS*20-1];
    logic signed [15:0] expected_membrane_before [0:NUM_CHECKPOINTS*64-1];
    logic signed [15:0] expected_membrane_after [0:NUM_CHECKPOINTS*64-1];
    logic signed [15:0] expected_membrane_reset [0:NUM_CHECKPOINTS*64-1];
    logic signed [31:0] expected_input_contribution [0:NUM_CHECKPOINTS*64-1];
    logic signed [31:0] expected_recurrent [0:NUM_CHECKPOINTS*64-1];
    logic expected_spikes [0:NUM_CHECKPOINTS*64-1];
    logic [4:0] expected_counts [0:NUM_CHECKPOINTS*64-1];
    logic [63:0] expected_previous_spikes [0:NUM_CHECKPOINTS-1];
    logic [4:0] expected_final_counts [0:NUM_SEGMENTS*64-1];

    integer segment;
    integer timestep;
    integer neuron;
    integer checkpoint_index;
    integer base_index;
    integer mismatches;
    integer membrane_before_comparisons;
    integer membrane_after_comparisons;
    integer membrane_reset_comparisons;
    integer input_comparisons;
    integer recurrent_comparisons;
    integer spike_comparisons;
    integer count_comparisons;
    integer previous_spike_comparisons;
    integer final_vector_comparisons;

    reservoir_controller dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .input_sample_valid(input_sample_valid),
        .input_sample(input_sample),
        .done(done),
        .spike_counts(spike_counts),
        .checkpoint_valid(checkpoint_valid),
        .checkpoint_timestep(checkpoint_timestep),
        .debug_state(debug_state),
        .debug_neuron_counter(debug_neuron_counter),
        .debug_timestep_counter(debug_timestep_counter),
        .checkpoint_membrane_before(checkpoint_membrane_before),
        .checkpoint_membrane_after(checkpoint_membrane_after),
        .checkpoint_membrane(checkpoint_membrane),
        .checkpoint_input(checkpoint_input),
        .checkpoint_recurrent(checkpoint_recurrent),
        .checkpoint_spikes(checkpoint_spikes),
        .checkpoint_spike_counts(checkpoint_spike_counts),
        .checkpoint_previous_spikes(checkpoint_previous_spikes)
    );

    always #5 clk = ~clk;

    task automatic fail_first_mismatch(input integer sample_id, input integer t, input integer n);
        begin
            $display("MISMATCH sample_id=%0d timestep=%0d logical_neuron=%0d input_q=%0d membrane_previous_expected=%0d membrane_previous_rtl=%0d previous_spike_vector=%016h recurrent_expected=%0d recurrent_rtl=%0d input_contribution_expected=%0d input_contribution_rtl=%0d membrane_after_expected=%0d membrane_after_rtl=%0d spike_expected=%0d spike_rtl=%0d spike_count_expected=%0d spike_count_rtl=%0d controller_state=%0d neuron_counter=%0d timestep_counter=%0d",
                sample_id, t, n, expected_input[sample_id*20+t],
                expected_membrane_before[checkpoint_index*64+n],
                $signed(checkpoint_membrane_before[n*16 +: 16]),
                checkpoint_previous_spikes,
                expected_recurrent[checkpoint_index*64+n],
                $signed(checkpoint_recurrent[n*32 +: 32]),
                expected_input_contribution[checkpoint_index*64+n],
                $signed(checkpoint_input[n*32 +: 32]),
                expected_membrane_after[checkpoint_index*64+n],
                $signed(checkpoint_membrane_after[n*16 +: 16]),
                expected_spikes[checkpoint_index*64+n], checkpoint_spikes[n],
                expected_counts[checkpoint_index*64+n],
                checkpoint_spike_counts[n*5 +: 5],
                debug_state, debug_neuron_counter, debug_timestep_counter);
            mismatches = mismatches + 1;
            $finish;
        end
    endtask

    task automatic compare_checkpoint(input integer sample_id, input integer t, input integer cp);
        begin
            checkpoint_index = cp;
            if (checkpoint_timestep !== t[4:0]) begin
                $display("MISMATCH checkpoint timestep sample_id=%0d expected=%0d rtl=%0d", sample_id, t, checkpoint_timestep);
                mismatches = mismatches + 1;
                $finish;
            end
            if (checkpoint_previous_spikes !== expected_previous_spikes[cp]) begin
                $display("MISMATCH previous_spikes sample_id=%0d timestep=%0d expected=%016h rtl=%016h", sample_id, t, expected_previous_spikes[cp], checkpoint_previous_spikes);
                mismatches = mismatches + 1;
                $finish;
            end
            previous_spike_comparisons = previous_spike_comparisons + 1;
            for (neuron = 0; neuron < 64; neuron = neuron + 1) begin
                if ($signed(checkpoint_membrane_before[neuron*16 +: 16]) !== $signed(expected_membrane_before[cp*64+neuron]))
                    fail_first_mismatch(sample_id, t, neuron);
                if ($signed(checkpoint_membrane_after[neuron*16 +: 16]) !== $signed(expected_membrane_after[cp*64+neuron]))
                    fail_first_mismatch(sample_id, t, neuron);
                if ($signed(checkpoint_membrane[neuron*16 +: 16]) !== $signed(expected_membrane_reset[cp*64+neuron]))
                    fail_first_mismatch(sample_id, t, neuron);
                if ($signed(checkpoint_input[neuron*32 +: 32]) !== $signed(expected_input_contribution[cp*64+neuron]))
                    fail_first_mismatch(sample_id, t, neuron);
                if ($signed(checkpoint_recurrent[neuron*32 +: 32]) !== $signed(expected_recurrent[cp*64+neuron]))
                    fail_first_mismatch(sample_id, t, neuron);
                if (checkpoint_spikes[neuron] !== expected_spikes[cp*64+neuron])
                    fail_first_mismatch(sample_id, t, neuron);
                if (checkpoint_spike_counts[neuron*5 +: 5] !== expected_counts[cp*64+neuron])
                    fail_first_mismatch(sample_id, t, neuron);
                if (checkpoint_spike_counts[neuron*5 +: 5] > 5'd20) begin
                    $display("MISMATCH spike count exceeded 20 sample_id=%0d timestep=%0d logical_neuron=%0d count=%0d", sample_id, t, neuron, checkpoint_spike_counts[neuron*5 +: 5]);
                    mismatches = mismatches + 1;
                    $finish;
                end
                membrane_before_comparisons = membrane_before_comparisons + 1;
                membrane_after_comparisons = membrane_after_comparisons + 1;
                membrane_reset_comparisons = membrane_reset_comparisons + 1;
                input_comparisons = input_comparisons + 1;
                recurrent_comparisons = recurrent_comparisons + 1;
                spike_comparisons = spike_comparisons + 1;
                count_comparisons = count_comparisons + 1;
            end
        end
    endtask

    task automatic send_sample(input integer input_index);
        begin
            @(negedge clk);
            input_sample_valid = 1'b1;
            input_sample = expected_input[input_index];
            @(posedge clk);
            #1 input_sample_valid = 1'b0;
        end
    endtask

    task automatic wait_for_checkpoint;
        begin
            while (!checkpoint_valid) begin
                @(posedge clk);
                #1;
            end
        end
    endtask

    initial begin
        $readmemh("rtl/mem/controller_inputs.mem", expected_input);
        $readmemh("rtl/mem/controller_expected_membrane_before.mem", expected_membrane_before);
        $readmemh("rtl/mem/controller_expected_membrane_after.mem", expected_membrane_after);
        $readmemh("rtl/mem/controller_expected_membrane_reset.mem", expected_membrane_reset);
        $readmemh("rtl/mem/controller_expected_input.mem", expected_input_contribution);
        $readmemh("rtl/mem/controller_expected_recurrent.mem", expected_recurrent);
        $readmemh("rtl/mem/controller_expected_spikes.mem", expected_spikes);
        $readmemh("rtl/mem/controller_expected_counts.mem", expected_counts);
        $readmemh("rtl/mem/controller_expected_previous_spikes.mem", expected_previous_spikes);
        $readmemh("rtl/mem/controller_expected_final_counts.mem", expected_final_counts);

        clk = 1'b0;
        reset = 1'b1;
        start = 1'b0;
        input_sample_valid = 1'b0;
        input_sample = 12'sd0;
        mismatches = 0;
        membrane_before_comparisons = 0;
        membrane_after_comparisons = 0;
        membrane_reset_comparisons = 0;
        input_comparisons = 0;
        recurrent_comparisons = 0;
        spike_comparisons = 0;
        count_comparisons = 0;
        previous_spike_comparisons = 0;
        final_vector_comparisons = 0;

        repeat (2) @(posedge clk);
        #1 reset = 1'b0;

        for (segment = 0; segment < NUM_SEGMENTS; segment = segment + 1) begin
            @(negedge clk);
            start = 1'b1;
            @(posedge clk);
            #1 start = 1'b0;

            for (timestep = 0; timestep < 20; timestep = timestep + 1) begin
                send_sample(segment*20+timestep);
                wait_for_checkpoint();
                compare_checkpoint(segment, timestep, segment*20+timestep);
            end

            if (!done) begin
                $display("MISMATCH missing done for sample_id=%0d", segment);
                mismatches = mismatches + 1;
                $finish;
            end
            for (neuron = 0; neuron < 64; neuron = neuron + 1) begin
                if (spike_counts[neuron*5 +: 5] !== expected_final_counts[segment*64+neuron]) begin
                    $display("MISMATCH final vector sample_id=%0d logical_neuron=%0d expected=%0d rtl=%0d", segment, neuron, expected_final_counts[segment*64+neuron], spike_counts[neuron*5 +: 5]);
                    mismatches = mismatches + 1;
                    $finish;
                end
                final_vector_comparisons = final_vector_comparisons + 1;
            end
            @(posedge clk);
            #1;
        end

        if (mismatches == 0) begin
            $display("PASS: controller checkpoints=%0d segments=%0d", NUM_CHECKPOINTS, NUM_SEGMENTS);
            $display("Membrane before comparisons: %0d", membrane_before_comparisons);
            $display("Membrane after comparisons: %0d", membrane_after_comparisons);
            $display("Membrane reset comparisons: %0d", membrane_reset_comparisons);
            $display("Input comparisons: %0d", input_comparisons);
            $display("Recurrent comparisons: %0d", recurrent_comparisons);
            $display("Spike comparisons: %0d", spike_comparisons);
            $display("Spike-count comparisons: %0d", count_comparisons);
            $display("Previous-spike-vector comparisons: %0d", previous_spike_comparisons);
            $display("Final-vector comparisons: %0d", final_vector_comparisons);
        end else begin
            $display("FAIL: controller mismatches=%0d", mismatches);
        end
        $finish;
    end
endmodule
