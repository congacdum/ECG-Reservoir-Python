module tb_sparse_recurrent_engine #(
    parameter integer NUM_VECTORS = 0,
    parameter integer EDGE_COUNT = 403
);
    logic [63:0] spike_vector_prev;
    logic [5:0] destination_neuron;
    logic signed [4:0] recurrent_sum;
    logic signed [31:0] recurrent_scaled_raw;
    logic signed [15:0] recurrent_contribution;

    logic [63:0] expected_spikes [0:NUM_VECTORS-1];
    logic [5:0] expected_destination [0:NUM_VECTORS-1];
    logic signed [4:0] expected_sum [0:NUM_VECTORS-1];
    logic signed [31:0] expected_raw [0:NUM_VECTORS-1];
    logic signed [15:0] expected_contribution [0:NUM_VECTORS-1];
    logic [63:0] expected_edge_mask [0:NUM_VECTORS-1];

    integer i;
    integer mismatches;
    integer comparisons;

    sparse_recurrent_engine #(.EDGE_COUNT(EDGE_COUNT)) dut (
        .spike_vector_prev(spike_vector_prev),
        .destination_neuron(destination_neuron),
        .recurrent_sum(recurrent_sum),
        .recurrent_scaled_raw(recurrent_scaled_raw),
        .recurrent_contribution(recurrent_contribution)
    );

    initial begin
        $readmemh("rtl/mem/recurrent_unit_spikes.mem", expected_spikes);
        $readmemh("rtl/mem/recurrent_unit_destination.mem", expected_destination);
        $readmemh("rtl/mem/recurrent_unit_expected_sum.mem", expected_sum);
        $readmemh("rtl/mem/recurrent_unit_expected_raw.mem", expected_raw);
        $readmemh("rtl/mem/recurrent_unit_expected_contribution.mem", expected_contribution);
        $readmemh("rtl/mem/recurrent_unit_expected_edge_mask.mem", expected_edge_mask);

        mismatches = 0;
        comparisons = 0;
        for (i = 0; i < NUM_VECTORS; i = i + 1) begin
            spike_vector_prev = expected_spikes[i];
            destination_neuron = expected_destination[i];
            #1;
            if ($signed(recurrent_sum) !== $signed(expected_sum[i]) ||
                $signed(recurrent_scaled_raw) !== $signed(expected_raw[i]) ||
                $signed(recurrent_contribution) !== $signed(expected_contribution[i])) begin
                $display("MISMATCH case=%0d destination_neuron=%0d previous_spike_vector=%016h active_sources=%016h expected_contributing_edges=%016h expected_raw_sum=%0d rtl_raw_sum=%0d expected_scaled_value=%0d rtl_scaled_value=%0d expected_contribution=%0d rtl_contribution=%0d",
                    i, destination_neuron, spike_vector_prev, spike_vector_prev,
                    expected_edge_mask[i], expected_sum[i], recurrent_sum,
                    expected_raw[i], recurrent_scaled_raw,
                    expected_contribution[i], recurrent_contribution);
                mismatches = mismatches + 1;
            end
            comparisons = comparisons + 3;
        end

        if (mismatches == 0)
            $display("PASS: %0d recurrent comparisons exact across %0d unit vectors", comparisons, NUM_VECTORS);
        else
            $display("FAIL: mismatches=%0d recurrent_comparisons=%0d", mismatches, comparisons);
        $finish;
    end
endmodule
