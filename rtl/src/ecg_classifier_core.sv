module ecg_classifier_core #(
    parameter integer NEURONS = 64,
    parameter integer TIMESTEPS = 20
)(
    input  logic clk,
    input  logic reset,
    input  logic start,
    input  logic input_sample_valid,
    input  logic signed [11:0] input_sample,
    output logic done,
    output logic signed [19:0] score,
    output logic predicted_class
);
    logic reservoir_done;
    logic [NEURONS*5-1:0] reservoir_counts;
    logic readout_start;
    logic readout_done;
    logic signed [19:0] readout_score;
    logic readout_class;

    reservoir_controller #(
        .NEURONS(NEURONS),
        .TIMESTEPS(TIMESTEPS)
    ) reservoir (
        .clk(clk),
        .reset(reset),
        .start(start),
        .input_sample_valid(input_sample_valid),
        .input_sample(input_sample),
        .done(reservoir_done),
        .spike_counts(reservoir_counts),
        .checkpoint_valid(),
        .checkpoint_timestep(),
        .debug_state(),
        .debug_neuron_counter(),
        .debug_timestep_counter(),
        .checkpoint_membrane_before(),
        .checkpoint_membrane_after(),
        .checkpoint_membrane(),
        .checkpoint_input(),
        .checkpoint_recurrent(),
        .checkpoint_spikes(),
        .checkpoint_spike_counts(),
        .checkpoint_previous_spikes()
    );

    // The controller updates its final count vector on the same edge that it
    // asserts done. Delay the readout start by one clock so the committed
    // vector, rather than the previous sample's vector, is captured.
    always_ff @(posedge clk) begin
        if (reset)
            readout_start <= 1'b0;
        else
            readout_start <= reservoir_done;
    end

    readout_mac #(.NEURONS(NEURONS)) readout (
        .clk(clk),
        .reset(reset),
        .start(readout_start),
        .spike_counts(reservoir_counts),
        .done(readout_done),
        .score(readout_score),
        .predicted_class(readout_class),
        .mac_valid(),
        .mac_index(),
        .mac_count(),
        .mac_weight(),
        .mac_product(),
        .mac_accumulator_before(),
        .mac_accumulator_after()
    );

    assign done = readout_done;
    assign score = readout_score;
    assign predicted_class = readout_class;
endmodule
