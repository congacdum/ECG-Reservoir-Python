module lif_pe #(
    parameter logic signed [15:0] THRESHOLD_Q = 16'sd2560,
    parameter integer LEAK_SHIFT = 3
)(
    input  logic signed [15:0] membrane_in,
    input  logic signed [31:0] input_contribution,
    input  logic signed [31:0] recurrent_contribution,
    input  logic               reset_state,
    output logic signed [15:0] membrane_before,
    output logic signed [15:0] membrane_after,
    output logic               spike,
    output logic signed [15:0] membrane_reset
);
    import fixed_point_pkg::*;

    logic signed [31:0] leak_term;
    logic signed [31:0] membrane_leaked;
    logic signed [31:0] membrane_update_raw;

    always_comb begin
        membrane_before = reset_state ? 16'sd0 : membrane_in;
        leak_term = 32'sd0;
        membrane_leaked = 32'sd0;
        membrane_update_raw = 32'sd0;
        membrane_after = 16'sd0;
        spike = 1'b0;
        membrane_reset = 16'sd0;

        if (!reset_state) begin
            leak_term = $signed(membrane_in) >>> LEAK_SHIFT;
            membrane_leaked = $signed(membrane_in) - leak_term;
            membrane_update_raw = membrane_leaked + input_contribution + recurrent_contribution;
            membrane_after = saturate_membrane(membrane_update_raw);
            spike = ($signed(membrane_after) >= $signed(THRESHOLD_Q));
            membrane_reset = spike ? 16'sd0 : membrane_after;
        end
    end
endmodule
