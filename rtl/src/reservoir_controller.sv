module reservoir_controller #(
    parameter integer NEURONS = 64,
    parameter integer TIMESTEPS = 20
)(
    input  logic clk,
    input  logic reset,
    input  logic start,
    input  logic input_sample_valid,
    input  logic signed [11:0] input_sample,
    output logic done,
    output logic [NEURONS*5-1:0] spike_counts,
    output logic checkpoint_valid,
    output logic [4:0] checkpoint_timestep,
    output logic [2:0] debug_state,
    output logic [5:0] debug_neuron_counter,
    output logic [4:0] debug_timestep_counter,
    output logic [NEURONS*16-1:0] checkpoint_membrane_before,
    output logic [NEURONS*16-1:0] checkpoint_membrane_after,
    output logic [NEURONS*16-1:0] checkpoint_membrane,
    output logic [NEURONS*32-1:0] checkpoint_input,
    output logic [NEURONS*32-1:0] checkpoint_recurrent,
    output logic [NEURONS-1:0] checkpoint_spikes,
    output logic [NEURONS*5-1:0] checkpoint_spike_counts,
    output logic [63:0] checkpoint_previous_spikes
);
    // At timestep t, membrane_current and previous_spikes are the committed
    // state from t-1. Each neuron writes only next-state storage. The commit
    // occurs after neuron 63, preventing same-timestep state contamination.
    // Ordering: input[t] + committed state -> recurrent -> LIF -> spike/reset
    // -> counter update -> full commit -> next timestep.
    localparam integer EDGE_COUNT = 403;

    typedef enum logic [2:0] {
        IDLE = 3'd0,
        LOAD_SAMPLE = 3'd1,
        PROCESS_NEURON = 3'd2,
        COMMIT_TIMESTEP = 3'd3,
        DONE_STATE = 3'd4
    } state_t;

    state_t state;
    logic signed [11:0] sample_q;
    logic [4:0] timestep_counter;
    logic [5:0] neuron_counter;

    logic signed [15:0] membrane_current [0:NEURONS-1];
    logic signed [15:0] membrane_next [0:NEURONS-1];
    logic [63:0] previous_spikes;
    logic [63:0] next_spikes;
    logic [4:0] spike_counts_current [0:NEURONS-1];
    logic [4:0] spike_counts_next [0:NEURONS-1];

    logic signed [15:0] debug_membrane_before [0:NEURONS-1];
    logic signed [15:0] debug_membrane_after [0:NEURONS-1];
    logic signed [31:0] debug_input [0:NEURONS-1];
    logic signed [31:0] debug_recurrent [0:NEURONS-1];
    logic [63:0] debug_previous_spikes;

    logic [1:0] input_weight_code [0:NEURONS-1];
    integer input_x3;
    logic signed [31:0] current_input_contribution;
    logic signed [15:0] current_membrane;
    logic signed [4:0] step_recurrent_sum;
    logic signed [31:0] step_recurrent_raw;
    logic signed [31:0] step_recurrent;
    logic signed [15:0] step_membrane_before;
    logic signed [15:0] step_membrane_after;
    logic step_spike;
    logic signed [15:0] step_membrane_reset;
    integer reset_index;

    initial begin
        $readmemb("outputs/fpga/weights/w_in_codes.mem", input_weight_code);
    end

    always_comb begin
        current_membrane = membrane_current[neuron_counter];
        input_x3 = $signed(sample_q) + ($signed(sample_q) <<< 1);
        current_input_contribution = 32'sd0;
        case (input_weight_code[neuron_counter])
            2'b00: current_input_contribution = input_x3 >>> 2;
            2'b01: current_input_contribution = input_x3 >>> 1;
            2'b10: current_input_contribution = input_x3;
            default: current_input_contribution = 32'sd0;
        endcase
    end

    reservoir_step #(.EDGE_COUNT(EDGE_COUNT)) step (
        .spike_vector_prev(previous_spikes),
        .destination_neuron(neuron_counter),
        .membrane_in(current_membrane),
        .input_contribution(current_input_contribution),
        .reset_state(1'b0),
        .recurrent_sum(step_recurrent_sum),
        .recurrent_scaled_raw(step_recurrent_raw),
        .recurrent_contribution(step_recurrent),
        .membrane_before(step_membrane_before),
        .membrane_after(step_membrane_after),
        .spike(step_spike),
        .membrane_reset(step_membrane_reset)
    );

    assign debug_state = state;
    assign debug_neuron_counter = neuron_counter;
    assign debug_timestep_counter = timestep_counter;

    genvar output_index;
    generate
        for (output_index = 0; output_index < NEURONS; output_index = output_index + 1) begin : PACKED_OUTPUTS
            assign spike_counts[output_index*5 +: 5] = spike_counts_current[output_index];
            assign checkpoint_membrane_before[output_index*16 +: 16] = debug_membrane_before[output_index];
            assign checkpoint_membrane_after[output_index*16 +: 16] = debug_membrane_after[output_index];
            assign checkpoint_membrane[output_index*16 +: 16] = membrane_current[output_index];
            assign checkpoint_input[output_index*32 +: 32] = debug_input[output_index];
            assign checkpoint_recurrent[output_index*32 +: 32] = debug_recurrent[output_index];
            assign checkpoint_spike_counts[output_index*5 +: 5] = spike_counts_current[output_index];
        end
    endgenerate

    assign checkpoint_spikes = previous_spikes;
    assign checkpoint_previous_spikes = debug_previous_spikes;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            sample_q <= 12'sd0;
            timestep_counter <= 5'd0;
            neuron_counter <= 6'd0;
            previous_spikes <= 64'd0;
            next_spikes <= 64'd0;
            debug_previous_spikes <= 64'd0;
            done <= 1'b0;
            checkpoint_valid <= 1'b0;
            checkpoint_timestep <= 5'd0;
            for (reset_index = 0; reset_index < NEURONS; reset_index = reset_index + 1) begin
                membrane_current[reset_index] <= 16'sd0;
                membrane_next[reset_index] <= 16'sd0;
                spike_counts_current[reset_index] <= 5'd0;
                spike_counts_next[reset_index] <= 5'd0;
                debug_membrane_before[reset_index] <= 16'sd0;
                debug_membrane_after[reset_index] <= 16'sd0;
                debug_input[reset_index] <= 32'sd0;
                debug_recurrent[reset_index] <= 32'sd0;
            end
        end else begin
            done <= 1'b0;
            checkpoint_valid <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        timestep_counter <= 5'd0;
                        neuron_counter <= 6'd0;
                        previous_spikes <= 64'd0;
                        next_spikes <= 64'd0;
                        debug_previous_spikes <= 64'd0;
                        for (reset_index = 0; reset_index < NEURONS; reset_index = reset_index + 1) begin
                            membrane_current[reset_index] <= 16'sd0;
                            membrane_next[reset_index] <= 16'sd0;
                            spike_counts_current[reset_index] <= 5'd0;
                            spike_counts_next[reset_index] <= 5'd0;
                            debug_membrane_before[reset_index] <= 16'sd0;
                            debug_membrane_after[reset_index] <= 16'sd0;
                            debug_input[reset_index] <= 32'sd0;
                            debug_recurrent[reset_index] <= 32'sd0;
                        end
                        state <= LOAD_SAMPLE;
                    end
                end

                LOAD_SAMPLE: begin
                    if (input_sample_valid) begin
                        sample_q <= input_sample;
                        debug_previous_spikes <= previous_spikes;
                        next_spikes <= 64'd0;
                        neuron_counter <= 6'd0;
                        state <= PROCESS_NEURON;
                    end
                end

                PROCESS_NEURON: begin
                    membrane_next[neuron_counter] <= step_membrane_reset;
                    next_spikes[neuron_counter] <= step_spike;
                    spike_counts_next[neuron_counter] <=
                        spike_counts_current[neuron_counter] + (step_spike ? 5'd1 : 5'd0);
                    debug_membrane_before[neuron_counter] <= step_membrane_before;
                    debug_membrane_after[neuron_counter] <= step_membrane_after;
                    debug_input[neuron_counter] <= current_input_contribution;
                    debug_recurrent[neuron_counter] <= step_recurrent;

                    if (neuron_counter == NEURONS - 1) begin
                        state <= COMMIT_TIMESTEP;
                    end else begin
                        neuron_counter <= neuron_counter + 6'd1;
                    end
                end

                COMMIT_TIMESTEP: begin
                    for (reset_index = 0; reset_index < NEURONS; reset_index = reset_index + 1) begin
                        membrane_current[reset_index] <= membrane_next[reset_index];
                        spike_counts_current[reset_index] <= spike_counts_next[reset_index];
                    end
                    previous_spikes <= next_spikes;
                    checkpoint_timestep <= timestep_counter;
                    checkpoint_valid <= 1'b1;

                    if (timestep_counter == TIMESTEPS - 1) begin
                        done <= 1'b1;
                        state <= DONE_STATE;
                    end else begin
                        timestep_counter <= timestep_counter + 5'd1;
                        state <= LOAD_SAMPLE;
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
