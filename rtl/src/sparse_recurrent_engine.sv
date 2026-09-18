module sparse_recurrent_engine #(
    parameter integer EDGE_COUNT = 403,
    parameter integer RECURRENT_SUM_BITS = 5,
    parameter integer RECURRENT_BITS = 16,
    parameter integer MEMBRANE_FRACTIONAL_BITS = 10,
    parameter integer RECURRENT_GAIN_SHIFT = 4
)(
    input  logic [63:0] spike_vector_prev,
    input  logic [5:0]  destination_neuron,
    output logic signed [RECURRENT_SUM_BITS-1:0] recurrent_sum,
    output logic signed [31:0] recurrent_scaled_raw,
    output logic signed [RECURRENT_BITS-1:0] recurrent_contribution
);
    // Frozen logical graph statistics: maximum incoming fan-in is 12.
    // Therefore the ternary edge sum is bounded by [-12, +12], which fits
    // in five signed bits (range [-16, +15]) without accumulator overflow.
    import fixed_point_pkg::*;

    logic [5:0] edge_source [0:EDGE_COUNT-1];
    logic [5:0] edge_destination [0:EDGE_COUNT-1];
    logic       edge_sign [0:EDGE_COUNT-1];
    integer edge_index;
    integer sum_work;
    integer scaled_work;

    initial begin
        $readmemh("rtl/mem/recurrent_edge_source.mem", edge_source);
        $readmemh("rtl/mem/recurrent_edge_destination.mem", edge_destination);
        $readmemh("rtl/mem/recurrent_edge_sign.mem", edge_sign);
    end

    always_comb begin
        sum_work = 0;
        for (edge_index = 0; edge_index < EDGE_COUNT; edge_index = edge_index + 1) begin
            if ((edge_destination[edge_index] == destination_neuron) &&
                spike_vector_prev[edge_source[edge_index]]) begin
                if (edge_sign[edge_index])
                    sum_work = sum_work - 1;
                else
                    sum_work = sum_work + 1;
            end
        end

        recurrent_sum = sum_work;
        scaled_work = sum_work *
            (1 <<< (MEMBRANE_FRACTIONAL_BITS - RECURRENT_GAIN_SHIFT));
        recurrent_scaled_raw = scaled_work;
        recurrent_contribution = saturate_membrane(scaled_work);
    end
endmodule
