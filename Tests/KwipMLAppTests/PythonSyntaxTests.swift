import XCTest
@testable import KwipMLApp

final class PythonSyntaxTests: XCTestCase {
    func testKeywordsNumbersCommentsAndCalls() {
        let source="def forward(self, x):\n    return torch.relu(x) + 1.25 # measured\n"
        let tokens=PythonSyntax.tokens(in:source)
        func values(_ kind:PythonTokenKind)->[String] { tokens.filter {$0.kind == kind}.map {(source as NSString).substring(with:$0.range)} }
        XCTAssertEqual(values(.keyword),["def","return"])
        XCTAssertEqual(values(.number),["1.25"])
        XCTAssertEqual(values(.comment),["# measured"])
        XCTAssertTrue(values(.name).contains("forward"));XCTAssertTrue(values(.name).contains("relu"))
    }
    func testStringsProtectHashAndKeywordsIncludingEscapesAndUnicode() {
        let source="label = 'café # return'\ntext = \"say \\\"if\\\"\"\n# class untouched\n"
        let tokens=PythonSyntax.tokens(in:source)
        let strings=tokens.filter {$0.kind == .string}.map {(source as NSString).substring(with:$0.range)}
        XCTAssertEqual(strings,["'café # return'","\"say \\\"if\\\"\""])
        XCTAssertEqual(tokens.filter {$0.kind == .keyword}.count,0)
        XCTAssertEqual(tokens.filter {$0.kind == .comment}.count,1)
        for token in tokens {XCTAssertLessThanOrEqual(NSMaxRange(token.range),(source as NSString).length)}
    }
    func testMultilineStringsAndKeywordBoundaries() {
        let source="\"\"\"class\n# docstring\"\"\"\nclassification = True\n"
        let tokens=PythonSyntax.tokens(in:source)
        XCTAssertEqual(tokens.filter {$0.kind == .string}.count,1)
        XCTAssertEqual(tokens.filter {$0.kind == .comment}.count,0)
        XCTAssertEqual(tokens.filter {$0.kind == .keyword}.map {(source as NSString).substring(with:$0.range)},["True"])
        XCTAssertTrue(PythonSyntax.tokens(in:"").isEmpty)
    }
}
