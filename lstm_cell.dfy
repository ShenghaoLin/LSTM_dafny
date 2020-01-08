method weight_multiplication(w: seq<seq<real>>, x: seq<real>) returns (y: seq<real>) 
    requires valid_wieght(w,x);
    ensures |w| == |y|;
 {
    var t := new real[|w|];
    var i := 0;
    var j := 0;
    while i < |w| 
        decreases |w| - i;
        invariant 0 <= i <= |w|
    {
        var j := 0;
        var s := 0.0;

        assert |w[i]| == |w[0]|;
        assert |w[i]| == |x|;
        while j < |w[i]| 
            decreases |w[i]| - j;
            invariant 0 <= j <= |w[i]|
        {
            s := s + w[i][j] * x[j];
            j := j + 1;
        }
        t[i] := s;
        i := i + 1;
    }
    y := t[..];
}

method vector_addiction(x: seq<real>, b: seq<real>) returns (y: seq<real>) 
    requires |x| == |b|;
    ensures |y| == |x|;
    ensures forall i :: 0 <= i < |x| ==> y[i] == x[i] + b[i]; 
{
    var t := new real[|x|];
    var i := 0;
    while i < |x|
        decreases |x| - i;
        invariant 0 <= i <= |x|;
        invariant forall j :: 0 <= j < i ==> t[j] == x[j] + b[j];
    {
        t[i] := x[i] + b[i];
        i := i + 1;
    }
    y := t[..];
}

method vector_multiplication(x_1: seq<real>, x_2: seq<real>) returns (y: seq<real>) 
    requires |x_1| == |x_2|;
    ensures |y| == |x_1|;
    ensures forall i :: 0 <= i < |x_1| ==> y[i] == x_1[i] * x_2[i]; 
{
    var t := new real[|x_1|];
    var i := 0;
    while i < |x_1|
        decreases |x_1| - i;
        invariant 0 <= i <= |x_1|;
        invariant forall j :: 0 <= j < i ==> t[j] == x_1[j] * x_2[j];
    {
        t[i] := x_1[i] * x_2[i];
        i := i + 1;
    }
    y := t[..];
}

predicate valid_wieght(w: seq<seq<real>>, x: seq<real>) {
    |w| > 0 && |w[0]| > 0 && |w[0]| == |x|
    && forall i :: 0 <= i < |w| ==> |w[i]| == |w[0]|
}

predicate valid_bias(w: seq<seq<real>>, b: seq<real>) {
    |w| == |b|
}

method e_power(x: real) returns (y: real) 
    ensures y > 0.0;
{
    // This is an approximation of e^x, given dafny is 
    // not very good at doing this kind of math
    var xx := if x >= 0.0 then x else -x;
    var ep := 1.0;
    var tmp := 1.0;
    var i := 1;
    var ii := 1.0;
    while i < 1000
        decreases 1000 - i;
        invariant ep > 0.0;
    {
        tmp := tmp * xx / ii;
        assert tmp >= 0.0;
        ep := ep + tmp;
        ii := ii + 1.0;
        i := i + 1;
    }
    y := if x >= 0.0 then ep else 1.0 / ep;
}

method sigmoid(x: real) returns (y: real) 
{
    var ep := e_power(-x);
    y := 1.0 / (1.0 + ep);
}

method tanh(x: real) returns (y: real)
{
    var epp := e_power(x);
    var epn := e_power(-x);
    y := (epp - epn) / (epp + epn);
} 

method vector_sigmoid(x: seq<real>) returns (y: seq<real>) 
    ensures |y| == |x|;
{
    var t := new real[|x|];
    var i:= 0;
    while i < |x|
        decreases |x| - i;
        invariant 0 <= i <= |x|;
    {
        t[i] := sigmoid(x[i]);
        i := i + 1;
    }
    y := t[..];
}

method vector_tanh(x: seq<real>) returns (y: seq<real>) 
    ensures |y| == |x|;
{
    var t := new real[|x|];
    var i:= 0;
    while i < |x|
        decreases |x| - i;
        invariant 0 <= i <= |x|;
    {
        t[i] := tanh(x[i]);
        i := i + 1;
    }
    y := t[..];
}

method sigmoid_with_weight(w: seq<seq<real>>, x: seq<real>, b: seq<real>) returns (y: seq<real>)
    requires valid_wieght(w, x);
    requires valid_bias(w, b);
    ensures |b| == |y|;
{
    var f := weight_multiplication(w, x);
    assert (|w| == |f|);
    f := vector_addiction(f , b);
    y := vector_sigmoid(f);
}

method tanh_with_weight(w: seq<seq<real>>, x: seq<real>, b: seq<real>) returns (y: seq<real>)
    requires valid_wieght(w, x);
    requires valid_bias(w, b);
    ensures |b| == |y|;
{
    var f := weight_multiplication(w, x);
    assert (|w| == |f|);
    f := vector_addiction(f , b);
    y := vector_tanh(f);
}
method lstm_cell(x: seq<real>, 
                 h_prev: seq<real>, 
                 c_prev: seq<real>,
                 w_f: seq<seq<real>>,
                 b_f: seq<real>,
                 w_i: seq<seq<real>>,
                 b_i: seq<real>,
                 w_c: seq<seq<real>>,
                 b_c: seq<real>,
                 w_o: seq<seq<real>>,
                 b_o: seq<real>) 
       returns (c: seq<real>, h: seq<real>)
    requires valid_wieght(w_f, h_prev + x);
    requires valid_bias(w_f, b_f);
    requires valid_wieght(w_i, h_prev + x);
    requires valid_bias(w_i, b_i);
    requires valid_wieght(w_c, h_prev + x);
    requires valid_bias(w_c, b_c);
    requires valid_wieght(w_o, h_prev + x);
    requires valid_bias(w_o, b_o);
    requires |c_prev| == |h_prev|;
    requires |b_i| == |c_prev|;
    requires |b_c| == |b_i|;
    requires |b_f| == |b_c|;
    requires |b_o| == |b_f|;
    ensures |h| == |h_prev|;
    ensures |c| == |c_prev|;
{
    var f_t := sigmoid_with_weight(w_f, h_prev + x, b_f);
    var i_t := sigmoid_with_weight(w_i, h_prev + x, b_i);    
    var c_t := tanh_with_weight(w_c, h_prev + x, b_c);
    c_t := vector_multiplication(c_t, i_t);
    var c_temp := vector_multiplication(c_prev, f_t);
    c := vector_addiction(c_temp, c_t);

    var o_t := sigmoid_with_weight(w_o, h_prev + x, b_o);
    var tanh_c := vector_tanh(c);
    h := vector_multiplication(o_t, tanh_c);
}