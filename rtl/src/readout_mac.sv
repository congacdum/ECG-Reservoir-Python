module readout_mac #(
    parameter integer NEURONS = 64,
    parameter integer COUNT_BITS = 5,
    parameter integer WEIGHT_BITS = 8,
    parameter integer ACCUMULATOR_BITS = 20
)(
    input  logic clk,
    input  logic reset,
    input  logic start,
    input  logic [NEURONS*COUNT_BITS-1:0] spike_counts,
    output logic done,
    output logic signed [ACCUMULATOR_BITS-1:0] score,
    output logic predicted_class,
    output logic mac_valid,
    output logic [6:0] mac_index,
    output logic [COUNT_BITS-1:0] mac_count,
    output logic signed [WEIGHT_BITS-1:0] mac_weight,
    output logic signed [31:0] mac_product,
    output logic signed [ACCUMULATOR_BITS-1:0] mac_accumulator_before,
    output logic signed [ACCUMULATOR_BITS-1:0] mac_accumulator_after
);
    localparam integer SCORE_MAX = (1 <<< (ACCUMULATOR_BITS - 1)) - 1;
    localparam integer SCORE_MIN = -(1 <<< (ACCUMULATOR_BITS - 1));

    typedef enum logic [1:0] {
        IDLE = 2'd0,
        MAC = 2'd1,
        DONE_STATE = 2'd2
    } state_t;

    state_t state;
    logic signed [WEIGHT_BITS-1:0] weight_rom [0:NEURONS-1];
    logic [6:0] neuron_index;
    logic [NEURONS*COUNT_BITS-1:0] counts_latched;
    logic signed [ACCUMULATOR_BITS-1:0] accumulator;
    logic [COUNT_BITS-1:0] current_count;
    logic signed [WEIGHT_BITS-1:0] current_weight;
    logic signed [31:0] count_extended;
    logic signed [31:0] product_extended;
    logic signed [31:0] accumulator_sum;
    logic signed [ACCUMULATOR_BITS-1:0] accumulator_next;
    integer reset_index;

    initial begin
        $readmemh("outputs/fpga/weights/w_out_int8.mem", weight_rom);
    end

    function automatic signed [ACCUMULATOR_BITS-1:0] saturate_score(
        input logic signed [31:0] value
    );
        begin
            if (value > SCORE_MAX)
                saturate_score = SCORE_MAX;
            else if (value < SCORE_MIN)
                saturate_score = SCORE_MIN;
            else
                saturate_score = value[ACCUMULATOR_BITS-1:0];
        end
    endfunction

    always_comb begin
        current_count = counts_latched[neuron_index*COUNT_BITS +: COUNT_BITS];
        current_weight = weight_rom[neuron_index];
        // The leading zero makes the unsigned count a positive signed value
        // before multiplication. The signed weight retains its INT8 sign.
        count_extended = $signed({27'd0, current_count});
        product_extended = count_extended * $signed(current_weight);
        accumulator_sum = $signed(accumulator) + product_extended;
        accumulator_next = saturate_score(accumulator_sum);
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            done <= 1'b0;
            score <= '0;
            predicted_class <= 1'b0;
            neuron_index <= 7'd0;
            counts_latched <= '0;
            accumulator <= '0;
            mac_valid <= 1'b0;
            mac_index <= 7'd0;
            mac_count <= '0;
            mac_weight <= '0;
            mac_product <= '0;
            mac_accumulator_before <= '0;
            mac_accumulator_after <= '0;
        end else begin
            done <= 1'b0;
            mac_valid <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        counts_latched <= spike_counts;
                        neuron_index <= 7'd0;
                        accumulator <= '0;
                        score <= '0;
                        predicted_class <= 1'b0;
                        state <= MAC;
                    end
                end

                MAC: begin
                    mac_valid <= 1'b1;
                    mac_index <= neuron_index;
                    mac_count <= current_count;
                    mac_weight <= current_weight;
                    mac_product <= product_extended;
                    mac_accumulator_before <= accumulator;
                    mac_accumulator_after <= accumulator_next;
                    accumulator <= accumulator_next;

                    if (neuron_index == NEURONS - 1) begin
                        score <= accumulator_next;
                        predicted_class <= (accumulator_next >= 0);
                        done <= 1'b1;
                        state <= DONE_STATE;
                    end else begin
                        neuron_index <= neuron_index + 7'd1;
                    end
                end

                DONE_STATE: begin
                    state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
