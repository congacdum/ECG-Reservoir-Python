module tb_reservoir_step_smoke_min;
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

    reservoir_step dut (
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
        // Frozen edge 0 is source 4 -> destination 0 with positive sign.
        spike_vector_prev = 64'h0000_0000_0000_0010;
        destination_neuron = 6'd0;
        membrane_in = 16'sd1000;
        input_contribution = 32'sd100;
        reset_state = 1'b0;
        #1;
        if (recurrent_sum !== 5'sd1 ||
            recurrent_scaled_raw !== 32'sd64 ||
            recurrent_contribution !== 32'sd64 ||
            membrane_before !== 16'sd1000 ||
            membrane_after !== 16'sd1039 ||
            spike !== 1'b0 ||
            membrane_reset !== 16'sd1039) begin
            $display("FAIL: minimal reservoir-step vector mismatch");
            $finish;
        end
        $display("PASS: minimal reservoir-step vector exact");
        $finish;
    end
endmodule
