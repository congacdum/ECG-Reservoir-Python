module tb_reservoir_step #(
    parameter integer NUM_VECTORS = 0,
    parameter integer EDGE_COUNT = 403
);
    logic [63:0] spike_vector_prev;
    logic [5:0] destination_neuron;
    logic signed [15:0] membrane_in;
    logic signed [31:0] input_contribution;
    logic reset_state;
    logic signed [4:0] recurrent_sum;
    logic signed [31:0] recurrent_scaled_raw;
    logic signed [31:0] recurrent_contribution;
    logic signed [15:0] membrane_before;
    logic signed [15:0] membrane_after;
    logic spike;
    logic signed [15:0] membrane_reset;

    logic [63:0] expected_spikes [0:NUM_VECTORS-1];
    logic [5:0] expected_destination [0:NUM_VECTORS-1];
    logic signed [15:0] expected_membrane_before [0:NUM_VECTORS-1];
    logic signed [31:0] expected_input [0:NUM_VECTORS-1];
    logic signed [4:0] expected_sum [0:NUM_VECTORS-1];
    logic signed [31:0] expected_raw [0:NUM_VECTORS-1];
    logic signed [31:0] expected_recurrent [0:NUM_VECTORS-1];
    logic signed [15:0] expected_after [0:NUM_VECTORS-1];
    logic expected_spike [0:NUM_VECTORS-1];
    logic signed [15:0] expected_reset [0:NUM_VECTORS-1];
    logic [63:0] expected_edge_mask [0:NUM_VECTORS-1];

    integer i;
    integer mismatches;
    integer comparisons;

    reservoir_step #(.EDGE_COUNT(EDGE_COUNT)) dut (
        .spike_vector_prev(spike_vector_prev),
        .destination_neuron(destination_neuron),
        .membrane_in(membrane_in),
        .input_contribution(input_contribution),
        .reset_state(reset_state),
        .recurrent_sum(recurrent_sum),
        .recurrent_scaled_raw(recurrent_scaled_raw),
        .recurrent_contribution(recurrent_contribution),
        .membrane_before(membrane_before),
        .membrane_after(membrane_after),
        .spike(spike),
        .membrane_reset(membrane_reset)
    );

    initial begin
        $readmemh("rtl/mem/reservoir_step_spikes.mem", expected_spikes);
        $readmemh("rtl/mem/reservoir_step_destination.mem", expected_destination);
        $readmemh("rtl/mem/reservoir_step_membrane_before.mem", expected_membrane_before);
        $readmemh("rtl/mem/reservoir_step_input.mem", expected_input);
        $readmemh("rtl/mem/reservoir_step_expected_sum.mem", expected_sum);
        $readmemh("rtl/mem/reservoir_step_expected_raw.mem", expected_raw);
        $readmemh("rtl/mem/reservoir_step_expected_recurrent.mem", expected_recurrent);
        $readmemh("rtl/mem/reservoir_step_expected_after.mem", expected_after);
        $readmemh("rtl/mem/reservoir_step_expected_spike.mem", expected_spike);
        $readmemh("rtl/mem/reservoir_step_expected_reset.mem", expected_reset);
        $readmemh("rtl/mem/reservoir_step_expected_edge_mask.mem", expected_edge_mask);

        mismatches = 0;
        comparisons = 0;
        reset_state = 1'b0;
        for (i = 0; i < NUM_VECTORS; i = i + 1) begin
            spike_vector_prev = expected_spikes[i];
            destination_neuron = expected_destination[i];
            membrane_in = expected_membrane_before[i];
            input_contribution = expected_input[i];
            #1;
            if ($signed(recurrent_sum) !== $signed(expected_sum[i]) ||
                $signed(recurrent_scaled_raw) !== $signed(expected_raw[i]) ||
                $signed(recurrent_contribution) !== $signed(expected_recurrent[i]) ||
                $signed(membrane_before) !== $signed(expected_membrane_before[i]) ||
                $signed(membrane_after) !== $signed(expected_after[i]) ||
                spike !== expected_spike[i] ||
                $signed(membrane_reset) !== $signed(expected_reset[i])) begin
                $display("MISMATCH sample_id=%0d timestep=%0d destination_neuron=%0d previous_spike_vector=%016h active_source_vector=%016h expected_contributing_edges=%016h expected_raw_sum=%0d rtl_raw_sum=%0d expected_scaled_value=%0d rtl_scaled_value=%0d expected_recurrent=%0d rtl_recurrent=%0d expected_membrane_before=%0d rtl_membrane_before=%0d expected_input=%0d expected_after=%0d rtl_after=%0d expected_spike=%0d rtl_spike=%0d expected_reset=%0d rtl_reset=%0d",
                    i / 64 / 20, (i / 64) % 20, destination_neuron,
                    spike_vector_prev, spike_vector_prev, expected_edge_mask[i],
                    expected_sum[i], recurrent_sum, expected_raw[i], recurrent_scaled_raw,
                    expected_recurrent[i], recurrent_contribution,
                    expected_membrane_before[i], membrane_before, expected_input[i],
                    expected_after[i], membrane_after, expected_spike[i], spike,
                    expected_reset[i], membrane_reset);
                mismatches = mismatches + 1;
            end
            comparisons = comparisons + 7;
        end

        if (mismatches == 0)
            $display("PASS: %0d reservoir-step comparisons exact across %0d neuron vectors", comparisons, NUM_VECTORS);
        else
            $display("FAIL: mismatches=%0d reservoir_step_comparisons=%0d", mismatches, comparisons);
        $finish;
    end
endmodule
