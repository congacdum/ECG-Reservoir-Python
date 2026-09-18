module reservoir_step #(
    parameter integer EDGE_COUNT = 403
)(
    input  logic [63:0] spike_vector_prev,
    input  logic [5:0]  destination_neuron,
    input  logic signed [15:0] membrane_in,
    input  logic signed [31:0] input_contribution,
    input  logic reset_state,
    output logic signed [4:0] recurrent_sum,
    output logic signed [31:0] recurrent_scaled_raw,
    output logic signed [31:0] recurrent_contribution,
    output logic signed [15:0] membrane_before,
    output logic signed [15:0] membrane_after,
    output logic spike,
    output logic signed [15:0] membrane_reset
);
    logic signed [15:0] recurrent_contribution_16;

    sparse_recurrent_engine #(
        .EDGE_COUNT(EDGE_COUNT)
    ) recurrent_engine (
        .spike_vector_prev(spike_vector_prev),
        .destination_neuron(destination_neuron),
        .recurrent_sum(recurrent_sum),
        .recurrent_scaled_raw(recurrent_scaled_raw),
        .recurrent_contribution(recurrent_contribution_16)
    );

    always_comb begin
        recurrent_contribution = {{16{recurrent_contribution_16[15]}}, recurrent_contribution_16};
    end

    lif_pe lif (
        .membrane_in(membrane_in),
        .input_contribution(input_contribution),
        .recurrent_contribution(recurrent_contribution),
        .reset_state(reset_state),
        .membrane_before(membrane_before),
        .membrane_after(membrane_after),
        .spike(spike),
        .membrane_reset(membrane_reset)
    );
endmodule
