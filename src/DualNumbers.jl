module DualNumbers

# Exporting the type and the general N-th order differentiation function
export Dual, dual_diff_n

# 1. Dual Number Type Definition
# Using T<:Number allows compatibility with standard numeric types
struct Dual{T<:Number} <: Number # Define a numeric type for automatic differentiation
    val::T # The real part or the value of the function at a point
    der::T # The infinitesimal part representing the derivative
end # End of the struct definition

# Convenience outer constructor with promotion
Dual(val::T, der::S) where {T<:Number, S<:Number} = begin
    v, d = promote(val, der)
    Dual{typeof(v)}(v, d)
end

# 2. Operator Overloading
# Importing standard functions from Base to allow multiple dispatch for the Dual type
import Base: +, -, *, /, ^, sin, cos, exp, sqrt, log, tan, asin, acos,
             tanh, sinh, cosh, ==, <, <=, promote_rule, convert, zero, one, show

# --- Basic Utilities ---
zero(::Type{Dual{T}}) where {T<:Number} = Dual(zero(T), zero(T))
one(::Type{Dual{T}})  where {T<:Number} = Dual(one(T), zero(T))

show(io::IO, d::Dual) = print(io, "Dual(", d.val, ", ", d.der, ")")

# --- Arithmetic Operations ---
# Addition: (a + bε) + (c + dε) = (a+c) + (b+d)ε
+(d1::Dual, d2::Dual) = Dual(d1.val + d2.val, d1.der + d2.der) # Add two dual numbers
+(d::Dual, a::Number) = Dual(d.val + a, d.der) # Add a dual number and a scalar
+(a::Number, d::Dual) = Dual(a + d.val, d.der) # Add a scalar and a dual number

# Unary plus
+(d::Dual) = d

# Subtraction: (a + bε) - (c + dε) = (a-c) + (b-d)ε
-(d1::Dual, d2::Dual) = Dual(d1.val - d2.val, d1.der - d2.der) # Subtract two dual numbers
-(d::Dual, a::Number) = Dual(d.val - a, d.der) # Subtract a scalar from a dual
-(a::Number, d::Dual) = Dual(a - d.val, -d.der) # Subtract a dual from a scalar

# Unary minus
-(d::Dual) = Dual(-d.val, -d.der)

# Multiplication: (a + bε)(c + dε) = ac + (ad + bc)ε
# Follows the product rule: (uv)' = u'v + uv'
*(d1::Dual, d2::Dual) = Dual(d1.val * d2.val, d1.der * d2.val + d1.val * d2.der) # Multiply duals
*(d::Dual, a::Number) = Dual(d.val * a, d.der * a) # Scale a dual by a scalar
*(a::Number, d::Dual) = Dual(a * d.val, a * d.der) # Scale a scalar by a dual

# Division: Uses the quotient rule (u/v)' = (u'v - uv') / v^2
/(d1::Dual, d2::Dual) = Dual(d1.val / d2.val,
                             (d1.der * d2.val - d1.val * d2.der) / (d2.val^2)) # Divide duals
/(d::Dual, a::Number) = Dual(d.val / a, d.der / a) # Divide a dual by a scalar
/(a::Number, d::Dual) = Dual(a / d.val, -a * d.der / (d.val^2)) # Divide a scalar by a dual

# --- Power Operations ---
# Power rule for Dual base: (D^n)' = n * D^(n-1) * D'
^(d::Dual, n::Integer) = Dual(d.val^n, n * d.val^(n - 1) * d.der) # Dual raised to an integer power
^(d::Dual, n::Real)    = Dual(d.val^n, n * d.val^(n - 1) * d.der) # Dual raised to a real power

# Power rule for Real base: (a^D)' = log(a) * a^D * D'
^(a::Number, d::Dual) = Dual(a^d.val, log(a) * a^d.val * d.der) # Scalar raised to a dual power

# General dual-to-dual power: u(x)^v(x) = exp(v(x) * log(u(x)))
^(d1::Dual, d2::Dual) = exp(d2 * log(d1))

# --- Elementary Functions ---
# Derivative of sin(x) is cos(x)
sin(d::Dual)  = Dual(sin(d.val), cos(d.val) * d.der) # Implement dual sine function

# Derivative of cos(x) is -sin(x)
cos(d::Dual)  = Dual(cos(d.val), -sin(d.val) * d.der) # Implement dual cosine function

# Derivative of tan(x) is sec^2(x) = 1 / cos^2(x)
tan(d::Dual)  = Dual(tan(d.val), d.der / cos(d.val)^2) # Implement dual tangent function

# Derivative of exp(x) is exp(x)
exp(d::Dual)  = Dual(exp(d.val), exp(d.val) * d.der) # Implement dual exponential function

# Derivative of sqrt(x) is 1 / (2*sqrt(x))
sqrt(d::Dual) = Dual(sqrt(d.val), d.der / (2 * sqrt(d.val))) # Implement dual square root

# Derivative of log(x) is 1 / x
log(d::Dual)  = Dual(log(d.val), d.der / d.val) # Implement dual natural logarithm

# Derivative of asin(x) is 1 / sqrt(1 - x^2)
asin(d::Dual) = Dual(asin(d.val), d.der / sqrt(1 - d.val^2)) # Implement dual inverse sine

# Derivative of acos(x) is -1 / sqrt(1 - x^2)
acos(d::Dual) = Dual(acos(d.val), -d.der / sqrt(1 - d.val^2)) # Implement dual inverse cosine

# --- Hyperbolic Functions ---
# Derivative of sinh(x) is cosh(x)
sinh(d::Dual) = Dual(sinh(d.val), cosh(d.val) * d.der) # Implement hyperbolic sine

# Derivative of cosh(x) is sinh(x)
cosh(d::Dual) = Dual(cosh(d.val), sinh(d.val) * d.der) # Implement hyperbolic cosine

# Derivative of tanh(x) is 1 - tanh^2(x)
tanh(d::Dual) = Dual(tanh(d.val), (1 - tanh(d.val)^2) * d.der) # Implement hyperbolic tangent

# --- Comparison Operators ---
# Comparisons are performed based on the real value (val) part of the dual number
==(d1::Dual, d2::Dual) = d1.val == d2.val && d1.der == d2.der # Overload equality
<(d1::Dual, d2::Dual)  = d1.val < d2.val # Overload less than
<=(d1::Dual, d2::Dual) = d1.val <= d2.val # Overload less than or equal to

# Optional mixed comparisons with scalars
==(d::Dual, a::Number) = d.val == a && d.der == zero(d.der)
==(a::Number, d::Dual) = d == a
<(d::Dual, a::Number)  = d.val < a
<(a::Number, d::Dual)  = a < d.val
<=(d::Dual, a::Number) = d.val <= a
<=(a::Number, d::Dual) = a <= d.val

# --- Type Promotion and Conversion ---
# This allows Dual numbers to work seamlessly with standard numeric types in Julia
promote_rule(::Type{Dual{T}}, ::Type{S}) where {T<:Number, S<:Number} = Dual{promote_type(T, S)} # Define promotion rule
convert(::Type{Dual{T}}, x::Number) where {T<:Number} = Dual(convert(T, x), zero(T)) # Define conversion rule

# 3. Higher-Order Derivative Evaluation

# Helper function to compute the first derivative
function dual_diff_1(f, x) # Takes function and point
    return f(Dual(x, one(x))).der # Seed with derivative 1 and extract the derivative part
end # End function

# General function to compute the N-th order derivative
function dual_diff_n(f, x, n::Integer) # Takes function, point, and derivative order
    n < 0 && throw(ArgumentError("Derivative order n must be nonnegative."))
    n == 0 && return f(x)

    g = f # Start with the original function
    for _ in 1:n # Repeatedly differentiate
        g_prev = g
        g = y -> dual_diff_1(g_prev, y)
    end # End loop

    return g(x) # Return the final N-th order derivative
end # End function

end # End of module
