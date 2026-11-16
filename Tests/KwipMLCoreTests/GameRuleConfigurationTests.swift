import XCTest
@testable import KwipMLCore
final class GameRuleConfigurationTests:XCTestCase {
    func testBundledDefinitionsHaveDifferentStateAndActions()throws {
        let scalar=try GameRuleConfiguration.example("take-away"),grid=try GameRuleConfiguration.example("placement"),gravity=try GameRuleConfiguration.example("gravity")
        XCTAssertNotEqual(scalar["state"],grid["state"]);XCTAssertNotEqual(grid["actions"],gravity["actions"])
        for d in [scalar,grid,gravity] {let bytes=try JSONEncoder().encode(d);XCTAssertEqual(try GameRuleConfiguration.decode(bytes),d)}
    }
    func testPracticeCompilesWithoutSourceOrModel()throws {
        let doc=try GameRuleConfiguration.document();try WorkflowCodec.validate(doc)
        XCTAssertEqual(doc.nodes.count,3);XCTAssertEqual(doc.edges.count,2)
        let decoded=try JSONDecoder().decode(WorkflowDocument.self,from:JSONEncoder().encode(doc));XCTAssertEqual(doc,decoded)
    }
    func testInvalidVersionAndUnknownEnvelopeRefused()throws {
        var d=try GameRuleConfiguration.example("take-away");d["version"]=2
        XCTAssertThrowsError(try GameRuleConfiguration.decode(JSONEncoder().encode(d)))
        XCTAssertThrowsError(try GameRuleConfiguration.validate(["definition":.object(d),"script":.string("import os")]))
    }
    func testOverBudgetDefinitionRefused()throws {XCTAssertThrowsError(try GameRuleConfiguration.decode(Data(repeating:32,count:131073)))}
}
extension GameRuleConfigurationTests {
    func testDuplicateKeysRefusedInSourceAndEscapedKeys()throws {
        for text in ["{\"state\":{},\"state\":{}}", "{\"x\":1,\"\\u0078\":2}"] {XCTAssertThrowsError(try GameRuleConfiguration.decodeValue(Data(text.utf8)))}
        XCTAssertNoThrow(try GameRuleConfiguration.decodeValue(Data("{\"x\":\"not a key: {\\\"x\\\"}\",\"nested\":[{\"x\":1}]}".utf8)))
    }
}
