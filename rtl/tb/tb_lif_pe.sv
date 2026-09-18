module tb_lif_pe #(
    parameter integer NUM_VECTORS = 10240,
    parameter integer NUM_DIRECTED = 6,
    parameter integer TOTAL_VECTORS = NUM_VECTORS + NUM_DIRECTED
);
    logic signed [15:0] membrane_in;
    logic signed [31:0] input_contribution;
    logic signed [31:0] recurrent_contribution;
    logic reset_state;
    logic signed [15:0] membrane_before;
    logic signed [15:0] membrane_after;
    logic spike;
    logic signed [15:0] membrane_reset;

    logic signed [15:0] expected_before [0:TOTAL_VECTORS-1];
    logic signed [31:0] expected_input [0:TOTAL_VECTORS-1];
    logic signed [31:0] expected_recurrent [0:TOTAL_VECTORS-1];
    logic signed [15:0] expected_after [0:TOTAL_VECTORS-1];
    logic expected_spike [0:TOTAL_VECTORS-1];
    logic signed [15:0] expected_reset [0:TOTAL_VECTORS-1];

    integer i;
    integer mismatches;
    integer state_comparisons;

    lif_pe dut (
        .membrane_in(membrane_in),
        .input_contribution(input_contribution),
        .recurrent_contribution(recurrent_contribution),
        .reset_state(reset_state),
        .membrane_before(membrane_before),
        .membrane_after(membrane_after),
        .spike(spike),
        .membrane_reset(membrane_reset)
    );

    task automatic compare_vector(input integer index);
        begin
            if ($signed(membrane_before) !== $signed(expected_before[index])) begin
                $display("MISMATCH index=%0d field=before rtl=%0d expected=%0d", index, membrane_before, expected_before[index]);
                mismatches = mismatches + 1;
            end
            if ($signed(membrane_after) !== $signed(expected_after[index])) begin
                $display("MISMATCH index=%0d field=after rtl=%0d expected=%0d", index, membrane_after, expected_after[index]);
                mismatches = mismatches + 1;
            end
            if (spike !== expected_spike[index]) begin
                $display("MISMATCH index=%0d field=spike rtl=%0d expected=%0d", index, spike, expected_spike[index]);
                mismatches = mismatches + 1;
            end
            if ($signed(membrane_reset) !== $signed(expected_reset[index])) begin
                $display("MISMATCH index=%0d field=reset rtl=%0d expected=%0d", index, membrane_reset, expected_reset[index]);
                mismatches = mismatches + 1;
            end
            state_comparisons = state_comparisons + 4;
        end
    endtask

    initial begin
        $readmemh("rtl/mem/lif_pe_expected_before.mem", expected_before);
        $readmemh("rtl/mem/lif_pe_input_contribution.mem", expected_input);
        $readmemh("rtl/mem/lif_pe_recurrent_contribution.mem", expected_recurrent);
        $readmemh("rtl/mem/lif_pe_expected_after.mem", expected_after);
        $readmemh("rtl/mem/lif_pe_expected_spike.mem", expected_spike);
        $readmemh("rtl/mem/lif_pe_expected_reset.mem", expected_reset);

        mismatches = 0;
        state_comparisons = 0;
        reset_state = 1'b0;

        for (i = 0; i < TOTAL_VECTORS; i = i + 1) begin
            membrane_in = expected_before[i];
            input_contribution = expected_input[i];
            recurrent_contribution = expected_recurrent[i];
            #1;
            compare_vector(i);
        end

        reset_state = 1'b1;
        membrane_in = 16'sd1234;
        input_contribution = 32'sd5678;
        recurrent_contribution = -32'sd90;
        #1;
        if (membrane_before !== 16'sd0 || membrane_after !== 16'sd0 ||
            spike !== 1'b0 || membrane_reset !== 16'sd0) begin
            $display("MISMATCH reset input handling");
            mismatches = mismatches + 1;
        end

        if (mismatches == 0)
            $display("PASS: %0d state comparisons bit-exact across %0d timestep vectors", state_comparisons, TOTAL_VECTORS);
        else
            $display("FAIL: mismatches=%0d state_comparisons=%0d", mismatches, state_comparisons);
        $finish;
    end
endmodule
