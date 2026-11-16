import XCTest
@testable import KwipMLCore

final class CanvasWorkflowPortPresentationTests: XCTestCase {
    func testSavedCheckpointPortUsesTrainedModelIdentity() {
        let descriptor=CanvasPortDescriptor(kind:CanvasPortKind(rawValue:"frozenModel") ?? .unknown,direction:.input)
        XCTAssertEqual(descriptor.shortLabel,"Trained model")
        XCTAssertNotEqual(descriptor.glyph,.question)
        XCTAssertNotEqual(descriptor.glyph,CanvasPortDescriptor(kind:.modelDefinition,direction:.input).glyph)
        XCTAssertTrue(descriptor.measured.isEmpty)
    }
    func testSevenSemanticKindsHaveDistinctNonColorIdentityWithoutMeasurementClaims() {
        let kinds: [CanvasPortKind] = [.tensor,.table,.dataset,.fittedState,.modelDefinition,.trainingData,.runConfiguration]
        let descriptors = kinds.map { CanvasPortDescriptor(kind:$0,direction:.output) }
        XCTAssertEqual(Set(descriptors.map { $0.glyph.rawValue }).count,7)
        XCTAssertEqual(Set(descriptors.map(\.shortLabel)).count,7)
        XCTAssertTrue(descriptors.allSatisfy { $0.measured.isEmpty && $0.accessibilityDescription.contains($0.shortLabel) })
        XCTAssertEqual(descriptors[4].shortLabel,"Model definition")
        XCTAssertEqual(descriptors[5].shortLabel,"Training data mapping")
        XCTAssertEqual(descriptors[6].shortLabel,"Run configuration")
        XCTAssertFalse(descriptors.contains { $0.glyph == .question })
        XCTAssertEqual(CanvasPortDescriptor.unknown(direction:.input).glyph,.question)
    }
}
