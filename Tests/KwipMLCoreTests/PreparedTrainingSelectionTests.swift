import XCTest
@testable import KwipMLCore
final class PreparedTrainingSelectionTests:XCTestCase {
    private var selection:PreparedTrainingSelection {PreparedTrainingSelection(manifestPath:"/private/prepared/manifest.json",manifestSHA256:String(repeating:"a",count:64),graphDigest:String(repeating:"B",count:64),trainingNode:"train-transform",trainingPort:"table",validationNode:"validation-transform",validationPort:"table",featureColumns:["x","y"],targetColumn:"label")}
    func testPreparedSelectionRoundTripAndLegacyDefault()throws {
        try selection.validate()
        var graph=ExampleFlows.parallelClassifier;graph.execution=ExecutionSettings(preparedTraining:selection)
        XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self,from:JSONEncoder().encode(graph)),graph)
        let legacy=try JSONDecoder().decode(ExecutionSettings.self,from:Data(#"{"device":"cpu","epochs":5}"#.utf8))
        XCTAssertNil(legacy.preparedTraining)
    }
    func testOrderedMappingSupportsVariableSampleSizeWithinDataBounds()throws {
        for count in [1,3,128] {
            var value=selection;value.featureColumns=(0..<count).map {"column \($0)"}
            try value.validate()
            XCTAssertEqual(try JSONDecoder().decode(PreparedTrainingSelection.self,from:JSONEncoder().encode(value)),value)
        }
        var value=selection;value.featureColumns=(0..<129).map {"c\($0)"}
        XCTAssertThrowsError(try value.validate())
    }
    func testRejectInvalidMappingHashAndEndpoint()throws {
        var value=selection;value.featureColumns=["x","x"];XCTAssertThrowsError(try value.validate())
        value=selection;value.targetColumn="x";XCTAssertThrowsError(try value.validate())
        value=selection;value.featureColumns=[];XCTAssertThrowsError(try value.validate())
        value=selection;value.trainingPort="";XCTAssertThrowsError(try value.validate())
        value=selection;value.manifestSHA256=String(repeating:"z",count:64);XCTAssertThrowsError(try value.validate())
        value=selection;value.graphDigest="abc";XCTAssertThrowsError(try value.validate())
        value=selection;value.manifestPath="";XCTAssertThrowsError(try value.validate())
    }
}
