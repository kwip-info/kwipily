import XCTest
@testable import KwipMLApp

final class ConfigurationColumnChoicesTests:XCTestCase {
    func testCapturedChoicesKeepMissingSavedNameWithoutSilentRemap() {
        let columns=[DataColumnSchema(name:"body",dtype:"string"),DataColumnSchema(name:"image",dtype:"image"),DataColumnSchema(name:"score",dtype:"float32")]
        let choices=ConfigurationColumnChoices.values(current:"retired",columns:columns,allowed:["string","float32"])
        XCTAssertEqual(choices.map(\.id),["retired","body","score"])
        XCTAssertTrue(choices[0].title.contains("unavailable"))
        XCTAssertEqual(ConfigurationColumnChoices.values(current:"body",columns:columns,allowed:["string"]).map(\.id),["body"])
    }
}
