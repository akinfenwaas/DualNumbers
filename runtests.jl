#include("C:\\Users\\akinf\\OneDrive\\Lecture Materials\\PhD GaSoU\\2-2 Spring 2026\\Scientific_Computing\\DualNumbers\\src") # Load the code to be tested
using Test
using DualNumbers

@testset "DualNumbers.jl 50-Question Test Suite" begin

    # -----------------------------------------------------------
    # 1. Arithmetic Operations (8 Tests)
    # -----------------------------------------------------------
    @testset "Arithmetic Operations" begin
        x = Dual(2.0, 1.0)
        y = Dual(3.0, 2.0)

        @test x + y == Dual(5.0, 3.0)
        @test x - y == Dual(-1.0, -1.0)
        @test x * y == Dual(6.0, 7.0)
        @test x / y == Dual(2.0 / 3.0, (1.0 * 3.0 - 2.0 * 2.0) / 3.0^2)
        @test 2.0 + x == Dual(4.0, 1.0)
        @test 2.0 - x == Dual(0.0, -1.0)
        @test 2.0 * x == Dual(4.0, 2.0)
        @test 2.0 / x == Dual(1.0, -0.5)
    end

    # -----------------------------------------------------------
    # 2. Power Functions (6 Tests)
    # -----------------------------------------------------------
    @testset "Power Functions" begin
        x = 2.0

        f1(x) = x^3
        f2(x) = 3.0^x
        f3(x) = x^x
        f4(x) = (x^2 + 1)^3
        f5(x) = x^(-2)
        f6(x) = sqrt(x)^3

        @test isapprox(dual_diff_n(f1, x, 1), 3x^2, atol=1e-10)
        @test isapprox(dual_diff_n(f2, x, 1), log(3.0) * 3.0^x, atol=1e-10)
        @test isapprox(dual_diff_n(f3, x, 1), x^x * (log(x) + 1), atol=1e-10)
        @test isapprox(dual_diff_n(f4, x, 1), 6x * (x^2 + 1)^2, atol=1e-10)
        @test isapprox(dual_diff_n(f5, x, 1), -2x^(-3), atol=1e-10)
        @test isapprox(dual_diff_n(f6, x, 1), 3/2 * sqrt(x), atol=1e-10)
    end

    # -----------------------------------------------------------
    # 3. Trigonometric Functions (5 Tests)
    # -----------------------------------------------------------
    @testset "Trigonometric Functions" begin
        x = 0.8

        f1(x) = sin(x)
        f2(x) = cos(x)
        f3(x) = tan(x)
        f4(x) = sin(x)^2 + cos(x)^2
        f5(x) = sin(cos(x))

        @test isapprox(dual_diff_n(f1, x, 1), cos(x), atol=1e-10)
        @test isapprox(dual_diff_n(f2, x, 1), -sin(x), atol=1e-10)
        @test isapprox(dual_diff_n(f3, x, 1), 1 / cos(x)^2, atol=1e-10)
        @test isapprox(dual_diff_n(f4, x, 1), 0.0, atol=1e-10)
        @test isapprox(dual_diff_n(f5, x, 1), cos(cos(x)) * (-sin(x)), atol=1e-10)
    end

    # -----------------------------------------------------------
    # 4. Inverse Trigonometric Functions (4 Tests)
    # -----------------------------------------------------------
    @testset "Inverse Trigonometric Functions" begin
        x = 0.4

        f1(x) = asin(x)
        f2(x) = acos(x)
        f3(x) = asin(x) + acos(x)
        f4(x) = asin(x^2)

        @test isapprox(dual_diff_n(f1, x, 1), 1 / sqrt(1 - x^2), atol=1e-10)
        @test isapprox(dual_diff_n(f2, x, 1), -1 / sqrt(1 - x^2), atol=1e-10)
        @test isapprox(dual_diff_n(f3, x, 1), 0.0, atol=1e-10)
        @test isapprox(dual_diff_n(f4, x, 1), 2x / sqrt(1 - x^4), atol=1e-10)
    end

    # -----------------------------------------------------------
    # 5. Hyperbolic Functions (6 Tests)
    # -----------------------------------------------------------
    @testset "Hyperbolic Functions" begin
        x = 0.7

        f1(x) = sinh(x)
        f2(x) = cosh(x)
        f3(x) = tanh(x)
        f4(x) = cosh(x)^2 - sinh(x)^2
        f5(x) = sinh(cosh(x))
        f6(x) = tanh(sinh(x))

        @test isapprox(dual_diff_n(f1, x, 1), cosh(x), atol=1e-10)
        @test isapprox(dual_diff_n(f2, x, 1), sinh(x), atol=1e-10)
        @test isapprox(dual_diff_n(f3, x, 1), 1 - tanh(x)^2, atol=1e-10)
        @test isapprox(dual_diff_n(f4, x, 1), 0.0, atol=1e-10)
        @test isapprox(dual_diff_n(f5, x, 1), cosh(cosh(x)) * sinh(x), atol=1e-10)
        @test isapprox(dual_diff_n(f6, x, 1), (1 - tanh(sinh(x))^2) * cosh(x), atol=1e-10)
    end

    # -----------------------------------------------------------
    # 6. Exponential and Logarithmic Functions (5 Tests)
    # -----------------------------------------------------------
    @testset "Exponential and Logarithmic Functions" begin
        x = 2.0

        f1(x) = exp(x)
        f2(x) = log(x)
        f3(x) = x^3 * log(x)
        f4(x) = log(exp(x))
        f5(x) = exp(log(x))

        @test isapprox(dual_diff_n(f1, x, 1), exp(x), atol=1e-10)
        @test isapprox(dual_diff_n(f2, x, 1), 1 / x, atol=1e-10)
        @test isapprox(dual_diff_n(f3, x, 1), 3x^2 * log(x) + x^2, atol=1e-10)
        @test isapprox(dual_diff_n(f4, x, 1), 1.0, atol=1e-10)
        @test isapprox(dual_diff_n(f5, x, 1), 1.0, atol=1e-10)
    end

    # -----------------------------------------------------------
    # 7. Composite Functions (6 Tests)
    # -----------------------------------------------------------
    @testset "Composite Functions" begin
        x = 1.2

        f1(x) = exp(sin(x))
        f2(x) = log(x^2 + 1)
        f3(x) = sqrt(x + cos(x))
        f4(x) = tanh(exp(sin(x)))
        f5(x) = exp(log(x^2 + 3))
        f6(x) = sqrt(exp(x) + sin(x))

        @test isapprox(dual_diff_n(f1, x, 1), exp(sin(x)) * cos(x), atol=1e-10)
        @test isapprox(dual_diff_n(f2, x, 1), 2x / (x^2 + 1), atol=1e-10)
        @test isapprox(dual_diff_n(f3, x, 1), (1 - sin(x)) / (2 * sqrt(x + cos(x))), atol=1e-10)
        @test isapprox(dual_diff_n(f4, x, 1), (1 - tanh(exp(sin(x)))^2) * exp(sin(x)) * cos(x), atol=1e-10)
        @test isapprox(dual_diff_n(f5, x, 1), 2x, atol=1e-10)
        @test isapprox(dual_diff_n(f6, x, 1), (exp(x) + cos(x)) / (2 * sqrt(exp(x) + sin(x))), atol=1e-10)
    end

    # -----------------------------------------------------------
    # 8. Product and Quotient Rule Stress Tests (5 Tests)
    # -----------------------------------------------------------
    @testset "Product and Quotient Rule" begin
        x = 1.5

        f1(x) = exp(x) * sin(x)
        f2(x) = (x^2 + 1) / (x - 1)
        f3(x) = (sin(x) * cosh(x)) / log(x + 2)
        f4(x) = (exp(x) * tanh(x)) / sqrt(x + 1)
        f5(x) = (x * sin(x) * exp(x)) / (log(x + 2))

        expected_f1 = exp(x) * sin(x) + exp(x) * cos(x)
        expected_f2 = (2x * (x - 1) - (x^2 + 1)) / (x - 1)^2
        expected_f3 = ((cos(x) * cosh(x) + sin(x) * sinh(x)) * log(x + 2) -
                      (sin(x) * cosh(x)) * (1 / (x + 2))) / log(x + 2)^2
        expected_f4 = ((exp(x) * tanh(x) + exp(x) * (1 - tanh(x)^2)) * sqrt(x + 1) -
                      (exp(x) * tanh(x)) * (1 / (2 * sqrt(x + 1)))) / (x + 1)
        expected_f5 = ((sin(x) * exp(x) + x * cos(x) * exp(x) + x * sin(x) * exp(x)) * log(x + 2) -
                      x * sin(x) * exp(x) * (1 / (x + 2))) / log(x + 2)^2

        @test isapprox(dual_diff_n(f1, x, 1), expected_f1, atol=1e-10)
        @test isapprox(dual_diff_n(f2, x, 1), expected_f2, atol=1e-10)
        @test isapprox(dual_diff_n(f3, x, 1), expected_f3, atol=1e-10)
        @test isapprox(dual_diff_n(f4, x, 1), expected_f4, atol=1e-10)
        @test isapprox(dual_diff_n(f5, x, 1), expected_f5, atol=1e-10)
    end

    # -----------------------------------------------------------
    # 9. Higher-Order Derivatives (8 Tests)
    # -----------------------------------------------------------
    @testset "Higher-Order Derivatives" begin
        x = 1.0

        f1(x) = x^4
        f2(x) = sin(x)
        f3(x) = exp(x)
        f4(x) = sinh(x)
        f5(x) = cosh(x)

        @test isapprox(dual_diff_n(f1, x, 2), 12x^2, atol=1e-10)
        @test isapprox(dual_diff_n(f1, x, 3), 24x, atol=1e-10)
        @test isapprox(dual_diff_n(f1, x, 4), 24.0, atol=1e-10)

        @test isapprox(dual_diff_n(f2, x, 2), -sin(x), atol=1e-10)
        @test isapprox(dual_diff_n(f2, x, 3), -cos(x), atol=1e-10)

        @test isapprox(dual_diff_n(f3, x, 2), exp(x), atol=1e-10)
        @test isapprox(dual_diff_n(f4, x, 3), cosh(x), atol=1e-10)
        @test isapprox(dual_diff_n(f5, x, 4), cosh(x), atol=1e-10)
    end

    # -----------------------------------------------------------
    # 10. Functional Identities (5 Tests)
    # -----------------------------------------------------------
    @testset "Functional Identities" begin
        x = 0.6

        f1(x) = log(exp(x))
        f2(x) = exp(log(x))
        f3(x) = asin(x) + acos(x)
        f4(x) = sin(x)^2 + cos(x)^2
        f5(x) = cosh(x)^2 - sinh(x)^2

        @test isapprox(dual_diff_n(f1, x, 1), 1.0, atol=1e-10)
        @test isapprox(dual_diff_n(f2, x, 1), 1.0, atol=1e-10)
        @test isapprox(dual_diff_n(f3, x, 1), 0.0, atol=1e-10)
        @test isapprox(dual_diff_n(f4, x, 1), 0.0, atol=1e-10)
        @test isapprox(dual_diff_n(f5, x, 1), 0.0, atol=1e-10)
    end

    # -----------------------------------------------------------
    # 11. Finite Difference Cross-Validation (5 Tests)
    # -----------------------------------------------------------
    @testset "Finite Difference Cross-Validation" begin
        h = 1e-6
        central_diff(f, x) = (f(x + h) - f(x - h)) / (2h)

        f1(x) = exp(sin(x)) + x^3 * log(x + 2)
        f2(x) = sqrt(x + 3) * tanh(x) - cos(x)^2
        f3(x) = (x^2 + sin(x)) / exp(x)
        f4(x) = log(x^2 + 4) * cosh(x)
        f5(x) = tanh(exp(x)) + sqrt(x + 5)

        x1 = 0.7
        x2 = 1.1
        x3 = 0.9
        x4 = 0.5
        x5 = 0.8

        @test isapprox(dual_diff_n(f1, x1, 1), central_diff(f1, x1), atol=1e-6)
        @test isapprox(dual_diff_n(f2, x2, 1), central_diff(f2, x2), atol=1e-6)
        @test isapprox(dual_diff_n(f3, x3, 1), central_diff(f3, x3), atol=1e-6)
        @test isapprox(dual_diff_n(f4, x4, 1), central_diff(f4, x4), atol=1e-6)
        @test isapprox(dual_diff_n(f5, x5, 1), central_diff(f5, x5), atol=1e-6)
    end

    # -----------------------------------------------------------
    # 12. Type Promotion and Comparison Logic (4 Tests)
    # -----------------------------------------------------------
    @testset "Type Promotion and Comparison Logic" begin
        x = Dual(2.0, 1.0)
        y = Dual(2.0, 1.0)
        z = Dual(3.0, 0.5)

        @test convert(Dual{Float64}, 5.0) == Dual(5.0, 0.0)
        @test x == y
        @test x < z
        @test x <= y
    end

    # -----------------------------------------------------------
    # 13. Composite Problems (3 Tests)
    # -----------------------------------------------------------
    @testset "Composite Problems" begin
        x = 1.1

        f1(x) = exp(x * sin(x)) * log(x + sqrt(x^2 + 1))
        f2(x) = tanh(log(x^2 + 2)) + sqrt(exp(x) + cos(x))
        f3(x) = x^x + exp(sinh(x)) - log(cosh(x))

        expected_f1 =
            exp(x * sin(x)) * (sin(x) + x * cos(x)) * log(x + sqrt(x^2 + 1)) +
            exp(x * sin(x)) * ((1 + x / sqrt(x^2 + 1)) / (x + sqrt(x^2 + 1)))

        expected_f2 =
            (1 - tanh(log(x^2 + 2))^2) * (2x / (x^2 + 2)) +
            (exp(x) - sin(x)) / (2 * sqrt(exp(x) + cos(x)))

        expected_f3 =
            x^x * (log(x) + 1) +
            exp(sinh(x)) * cosh(x) -
            sinh(x) / cosh(x)

        @test isapprox(dual_diff_n(f1, x, 1), expected_f1, atol=1e-10)
        @test isapprox(dual_diff_n(f2, x, 1), expected_f2, atol=1e-10)
        @test isapprox(dual_diff_n(f3, x, 1), expected_f3, atol=1e-10)
    end

end




using Test
using DualNumbers

@testset "DualNumbers.jl Test Suite" begin

    @testset "Constructors and Basic Fields" begin
        d = Dual(3.0, 1.0)
        @test d.val == 3.0
        @test d.der == 1.0

        d2 = Dual(2, 1.0)
        @test d2.val == 2.0
        @test d2.der == 1.0
    end

    @testset "zero and one" begin
        z = zero(Dual{Float64})
        o = one(Dual{Float64})
        @test z == Dual(0.0, 0.0)
        @test o == Dual(1.0, 0.0)
    end

    @testset "Unary Operations" begin
        d = Dual(2.0, 3.0)
        @test +d == Dual(2.0, 3.0)
        @test -d == Dual(-2.0, -3.0)
    end

    @testset "Addition" begin
        d1 = Dual(2.0, 1.0)
        d2 = Dual(3.0, 4.0)
        @test d1 + d2 == Dual(5.0, 5.0)
        @test d1 + 5 == Dual(7.0, 1.0)
        @test 5 + d1 == Dual(7.0, 1.0)
    end

    @testset "Subtraction" begin
        d1 = Dual(5.0, 2.0)
        d2 = Dual(3.0, 1.0)
        @test d1 - d2 == Dual(2.0, 1.0)
        @test d1 - 2 == Dual(3.0, 2.0)
        @test 7 - d1 == Dual(2.0, -2.0)
    end

    @testset "Multiplication" begin
        d1 = Dual(2.0, 1.0)
        d2 = Dual(3.0, 4.0)
        @test d1 * d2 == Dual(6.0, 11.0)
        @test d1 * 5 == Dual(10.0, 5.0)
        @test 5 * d1 == Dual(10.0, 5.0)
    end

    @testset "Division" begin
        d1 = Dual(6.0, 2.0)
        d2 = Dual(3.0, 1.0)
        q = d1 / d2
        @test q.val ≈ 2.0
        @test q.der ≈ 0.0

        @test d1 / 2 == Dual(3.0, 1.0)

        q2 = 12 / Dual(3.0, 1.0)
        @test q2.val ≈ 4.0
        @test q2.der ≈ -4/3
    end

    @testset "Integer and Real Powers" begin
        d = Dual(2.0, 1.0)
        @test d^3 == Dual(8.0, 12.0)
        @test d^2.5 ≈ Dual(2.0^2.5, 2.5 * 2.0^1.5)
    end

    @testset "Scalar Base to Dual Exponent" begin
        d = Dual(2.0, 1.0)
        y = 3^d
        @test y.val ≈ 9.0
        @test y.der ≈ log(3) * 9.0
    end

    @testset "Dual to Dual Power" begin
        d1 = Dual(2.0, 1.0)
        d2 = Dual(3.0, 2.0)
        y = d1^d2
        expected_val = 2.0^3.0
        expected_der = expected_val * (2.0 * log(2.0) + 3.0 / 2.0)
        @test y.val ≈ expected_val
        @test y.der ≈ expected_der
    end

    @testset "Trigonometric Functions" begin
        d = Dual(pi/4, 1.0)

        s = sin(d)
        @test s.val ≈ sin(pi/4)
        @test s.der ≈ cos(pi/4)

        c = cos(d)
        @test c.val ≈ cos(pi/4)
        @test c.der ≈ -sin(pi/4)

        t = tan(d)
        @test t.val ≈ tan(pi/4)
        @test t.der ≈ 1 / cos(pi/4)^2
    end

    @testset "Exponential, Logarithm, and Square Root" begin
        d = Dual(4.0, 1.0)

        e = exp(d)
        @test e.val ≈ exp(4.0)
        @test e.der ≈ exp(4.0)

        l = log(d)
        @test l.val ≈ log(4.0)
        @test l.der ≈ 1/4

        s = sqrt(d)
        @test s.val ≈ 2.0
        @test s.der ≈ 1/4
    end

    @testset "Inverse Trigonometric Functions" begin
        d = Dual(0.5, 1.0)

        a1 = asin(d)
        @test a1.val ≈ asin(0.5)
        @test a1.der ≈ 1 / sqrt(1 - 0.5^2)

        a2 = acos(d)
        @test a2.val ≈ acos(0.5)
        @test a2.der ≈ -1 / sqrt(1 - 0.5^2)
    end

    @testset "Hyperbolic Functions" begin
        d = Dual(0.7, 1.0)

        sh = sinh(d)
        @test sh.val ≈ sinh(0.7)
        @test sh.der ≈ cosh(0.7)

        ch = cosh(d)
        @test ch.val ≈ cosh(0.7)
        @test ch.der ≈ sinh(0.7)

        th = tanh(d)
        @test th.val ≈ tanh(0.7)
        @test th.der ≈ 1 - tanh(0.7)^2
    end

    @testset "Comparisons" begin
        d1 = Dual(2.0, 1.0)
        d2 = Dual(2.0, 1.0)
        d3 = Dual(3.0, 0.0)

        @test d1 == d2
        @test d1 != d3
        @test d1 < d3
        @test d1 <= d2
        @test d1 <= d3

        @test Dual(2.0, 0.0) == 2.0
        @test 2.0 == Dual(2.0, 0.0)
        @test Dual(1.5, 8.0) < 2.0
        @test 1.0 < Dual(2.0, 9.0)
        @test Dual(2.0, 4.0) <= 2.0
    end

    @testset "Conversion and Promotion" begin
        d = convert(Dual{Float64}, 5)
        @test d == Dual(5.0, 0.0)

        x = Dual(2.0, 1.0) + 3
        @test x == Dual(5.0, 1.0)
    end

    @testset "First Derivative Checks" begin
        @test dual_diff_n(x -> x^2, 3.0, 1) ≈ 6.0
        @test dual_diff_n(x -> x^3, 2.0, 1) ≈ 12.0
        @test dual_diff_n(x -> 5x + 1, 10.0, 1) ≈ 5.0
        @test dual_diff_n(sin, 0.0, 1) ≈ 1.0
        @test dual_diff_n(cos, 0.0, 1) ≈ 0.0
        @test dual_diff_n(exp, 0.0, 1) ≈ 1.0
        @test dual_diff_n(log, 2.0, 1) ≈ 1/2
        @test dual_diff_n(sqrt, 4.0, 1) ≈ 1/4
    end

    @testset "Higher-Order Derivatives" begin
        f(x) = x^4
        @test dual_diff_n(f, 3.0, 0) ≈ 81.0
        @test dual_diff_n(f, 3.0, 1) ≈ 108.0
        @test dual_diff_n(f, 3.0, 2) ≈ 108.0
        @test dual_diff_n(f, 3.0, 3) ≈ 72.0
        @test dual_diff_n(f, 3.0, 4) ≈ 24.0
        @test dual_diff_n(f, 3.0, 5) ≈ 0.0
    end

    @testset "Composite Function Derivatives" begin
        f(x) = exp(sin(x) + x^2)
        x0 = 1.0
        expected = exp(sin(x0) + x0^2) * (cos(x0) + 2x0)
        @test dual_diff_n(f, x0, 1) ≈ expected
    end

    @testset "Polynomial with Known Higher Derivatives" begin
        f(x) = x^5 - 3x^4 + 2x^3 - x + 7
        x0 = 2.0

        f1 = 5x0^4 - 12x0^3 + 6x0^2 - 1
        f2 = 20x0^3 - 36x0^2 + 12x0
        f3 = 60x0^2 - 72x0 + 12
        f4 = 120x0 - 72
        f5 = 120.0

        @test dual_diff_n(f, x0, 1) ≈ f1
        @test dual_diff_n(f, x0, 2) ≈ f2
        @test dual_diff_n(f, x0, 3) ≈ f3
        @test dual_diff_n(f, x0, 4) ≈ f4
        @test dual_diff_n(f, x0, 5) ≈ f5
    end

    @testset "Constants and Zero Derivatives" begin
        @test dual_diff_n(x -> 5.0, 2.0, 1) ≈ 0.0
        @test dual_diff_n(x -> 5.0, 2.0, 2) ≈ 0.0
        @test dual_diff_n(x -> 5.0, 2.0, 3) ≈ 0.0
    end

    @testset "Argument Validation" begin
        @test_throws ArgumentError dual_diff_n(x -> x^2, 1.0, -1)
    end

end