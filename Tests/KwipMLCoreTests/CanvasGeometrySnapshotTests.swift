import XCTest
@testable import KwipMLCore

final class CanvasGeometrySnapshotTests: XCTestCase {
    func testOptInGeometryReadDiagnostic() throws {
        guard ProcessInfo.processInfo.environment["KWIPML_CANVAS_BENCHMARK"] == "1" else {
            throw XCTSkip("Opt-in diagnostic; no timing threshold")
        }
        let registry = ComponentRegistry(components: (0..<256).map {
            .init(id: "type-\($0)", inputs: Array(repeating: "in", count: $0 % 5), outputs: ["out"])
        })
        var settings = CanvasPresentationSettings()
        settings.flowDirection = .topToBottom; settings.nodeDetail = .large
        let start = Date()
        var baseline = 0.0
        for _ in 0..<8 {
            for definition in registry.components {
                let projection = CanvasCoordinateProjection(settings: settings,
                    maximumInputCount: registry.components.map { $0.inputs.count }.max() ?? 0,
                    maximumOutputCount: registry.components.map { $0.outputs.count }.max() ?? 0)
                let found = registry.component(type: definition.id)
                let size = CanvasNodeGeometry.size(detail: settings.nodeDetail, direction: settings.flowDirection,
                    inputCount: found?.inputs.count ?? 0, outputCount: found?.outputs.count ?? 0)
                baseline += projection.project(CGPoint(x: 75, y: 123)).x + size.height
            }
        }
        let baselineMS = Date().timeIntervalSince(start) * 1000
        let indexedStart = Date()
        let snapshot = CanvasGeometrySnapshot(registry: registry, presentation: settings)
        var indexed = 0.0
        for _ in 0..<8 {
            for definition in registry.components {
                indexed += snapshot.projection.project(CGPoint(x: 75, y: 123)).x + snapshot.size(type: definition.id).height
            }
        }
        let indexedMS = Date().timeIntervalSince(indexedStart) * 1000
        XCTAssertEqual(indexed, baseline)
        print("GEOMETRY_READ types=256 reads=2048 baseline_ms=\(baselineMS) snapshot_including_build_ms=\(indexedMS)")
    }

    func testSnapshotMatchesGeometryInEveryPresentationAndPreservesFirstDefinition() {
        let definitions: [ComponentDefinition] = [
            .init(id: "wide", inputs: ["a", "b", "c", "d"], outputs: ["x", "y"]),
            .init(id: "single", inputs: ["in"], outputs: ["out"]),
            .init(id: "wide", inputs: Array(repeating: "duplicate", count: 8))]
        let registry = ComponentRegistry(components: definitions)
        for direction in [CanvasFlowDirection.leftToRight, .topToBottom] {
            for detail in [CanvasNodeDetail.small, .medium, .large] {
                var settings = CanvasPresentationSettings()
                settings.flowDirection = direction; settings.nodeDetail = detail
                let snapshot = CanvasGeometrySnapshot(registry: registry, presentation: settings)
                let expected = CanvasCoordinateProjection(settings: settings, maximumInputCount: 8, maximumOutputCount: 2)
                XCTAssertEqual(snapshot.projection, expected)
                for type in ["wide", "single", "unknown"] {
                    let definition = registry.component(type: type)
                    XCTAssertEqual(snapshot.definition(type: type), definition)
                    XCTAssertEqual(snapshot.size(type: type), CanvasNodeGeometry.size(detail: detail, direction: direction,
                        inputCount: definition?.inputs.count ?? 0, outputCount: definition?.outputs.count ?? 0))
                }
                let point = CGPoint(x: -35, y: 219)
                XCTAssertEqual(snapshot.projection.project(point), expected.project(point))
                XCTAssertEqual(snapshot.projection.inverse(expected.project(point)).x, point.x, accuracy: 1e-10)
            }
        }
        let original = CanvasGeometrySnapshot(registry: registry, presentation: .init())
        let replaced = CanvasGeometrySnapshot(registry: .init(), presentation: .init())
        XCTAssertNotNil(original.definition(type: "wide"))
        XCTAssertNil(replaced.definition(type: "wide"))
    }
}
