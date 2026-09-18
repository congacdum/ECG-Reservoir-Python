package fixed_point_pkg;
    localparam integer MEMBRANE_BITS = 16;
    localparam integer LEAK_SHIFT = 3;
    localparam logic signed [15:0] THRESHOLD_Q = 16'sd2560;

    function automatic logic signed [15:0] saturate_membrane(
        input logic signed [31:0] value
    );
        begin
            if (value > 32'sd32767)
                saturate_membrane = 16'sh7fff;
            else if (value < -32'sd32768)
                saturate_membrane = 16'sh8000;
            else
                saturate_membrane = value[15:0];
        end
    endfunction
endpackage
