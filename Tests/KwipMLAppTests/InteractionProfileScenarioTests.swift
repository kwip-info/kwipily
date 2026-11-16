import XCTest
import SwiftUI
import AppKit
import KwipMLCore
@testable import KwipMLApp

/// Opt-in real editor + hosted SwiftUI scenarios. These measure command and view
/// work, not physical mouse latency or GPU presentation. No user project is opened.
@MainActor final class InteractionProfileScenarioTests: XCTestCase {
    func testScriptedCanvasProfile() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let output = env["KWIPML_PROFILE_OUTPUT"] else { throw XCTSkip("Run scripts/interaction_profile.py run") }
        let count = Int(env["KWIPML_PROFILE_NODES"] ?? "64") ?? 64
        let iterations = Int(env["KWIPML_PROFILE_ITERATIONS"] ?? "8") ?? 8
        guard [16,64,256].contains(count), (3...30).contains(iterations) else { throw NSError(domain: "Invalid profile fixture", code: 1) }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models = try ComponentRegistry.builtIn()
        let data = try DataFlowOperatorRegistry.load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
        let modelCount = min(count, 127) // Leave room for insertion within the model-domain limit.
        let addresses = (0..<count).map { index in
            WorkflowNodeAddress(domain: index < modelCount ? .model : (index == count-1 ? .control : .data), localID: "profile-\(index)")
        }
        let nodes = addresses.enumerated().map { index, address -> WorkflowNode in
            let payload: WorkflowNodePayload
            switch address.domain {
            case .control: payload = .modelDefinition
            case .data: payload = .dataOperator(operatorID: "kwip.data.sourceCSV", config: ["separator":.string(",")])
            case .model:
                payload = index == 0 ? .modelPrimitive(type: "input", configurationVersion: 1, config: ["features":2]) : .modelPrimitive(type: "linear", configurationVersion: 2, config: ["in_features":2,"out_features":2,"bias":true])
            }
            return WorkflowNode(address: address, name: "", x: Double(index % 8) * 260, y: Double(index / 8) * 180, payload: payload)
        }
        // Leave 0 -> 1 open for the connection scenario; retain other deterministic wires.
        let edges = (2..<modelCount).map { index in
            WorkflowEdge(id: "profile-edge-\(index)", from: .init(address: addresses[index-1], port: "out"), to: .init(address: addresses[index], port: "in"), kind: .tensor)
        }
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: nodes, edges: edges)
        try WorkflowCodec.validate(document)
        let library = ProjectLibrary(directory: root)
        _ = try library.saveSource(.workflow(document))
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID, modelRegistry: models, dataRegistry: data)
        let adapter = try WorkflowCanvasAdapter(owner: owner, modelRegistry: models, dataRegistry: data)
        defer { adapter.invalidate() }
        let first = try XCTUnwrap(adapter.projection.displayIDByAddress[addresses[0]])
        let second = try XCTUnwrap(adapter.projection.displayIDByAddress[addresses[1]])
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 1000, height: 700), styleMask: [.titled], backing: .buffered, defer: false)
        window.title = "Kwip ML · Interaction profiler fixture"
        window.isReleasedWhenClosed = false
        let host = NSHostingView(rootView: GraphCanvasView(store: adapter, publishesHistoryFocus: false))
        window.contentView = host
        window.orderFront(nil)
        defer { window.orderOut(nil); window.close() }
        func layout() { InteractionProfiler.shared.measure(.hostLayout) { host.layoutSubtreeIfNeeded(); host.displayIfNeeded() } }
        // Warm layout and initial viewport restore outside measurement.
        layout(); try await Task.sleep(for: .milliseconds(350)); layout()
        let profiler = InteractionProfiler.shared
        profiler.start(maximumSamples: 100_000)
        profiler.scenario = .idle
        let probe = InteractionMainQueueProbe()
        defer { probe.stop(); _ = profiler.stop() }
        func run(_ scenario: InteractionProfiler.Scenario, _ action: (Int) throws -> Void) async throws {
            profiler.scenario = scenario
            for index in 0..<iterations {
                try profiler.measure(.scenarioAction) { try action(index) }
                layout()
                try await Task.sleep(for: .milliseconds(25))
                layout()
            }
        }
        try await run(.idle) { _ in }
        try await run(.selection) { i in adapter.selectNode(i % 2 == 0 ? first : second, extending: false) }
        XCTAssertEqual(owner.selection.nodeAddresses.count, 1)
        try await run(.move) { i in adapter.moveNodes(ids: [second], dx: i % 2 == 0 ? 5 : -5, dy: 0) }
        XCTAssertEqual(owner.document.nodes[1].x, nodes[1].x + (iterations % 2 == 0 ? 0 : 5))
        try await run(.configuration) { i in
            XCTAssertTrue(adapter.configure(nodeID: second, config: ["in_features":2,"out_features":2,"bias":.bool(i % 2 != 0)], lease: owner.lease()))
        }
        try await run(.hover) { _ in
            let target = try XCTUnwrap(adapter.graph.nodes.first { $0.id == second })
            for _ in 0..<20 { XCTAssertTrue(adapter.canConnect(source: first, sourcePort: "out", target: target, port: "in")) }
        }
        try await run(.connection) { _ in
            adapter.connect(source: first, sourcePort: "out", target: second, targetPort: "in")
            XCTAssertEqual(owner.document.edges.count, edges.count + 1)
            adapter.undo(); XCTAssertEqual(owner.document.edges.count, edges.count)
        }
        try await run(.undoRedo) { _ in
            adapter.moveNodes(ids: [second], dx: 1, dy: 0)
            adapter.undo(); adapter.redo(); adapter.undo()
        }
        let entry = try XCTUnwrap(adapter.creationEntries.first { if case .modelPrimitive("linear", 2, _) = $0.payload { return true }; return false })
        try await run(.insertionDeletion) { _ in
            // At the overall256-node ceiling, free a slot and restore its exact
            // fixture node after the insertion/deletion cycle. Never raise limits.
            let displaced = count == 256 ? nodes.last : nil
            if let displaced {
                let id = try XCTUnwrap(adapter.projection.displayIDByAddress[displaced.address])
                adapter.selectNode(id, extending: false); adapter.deleteSelection()
            }
            let baseCount = count - (displaced == nil ? 0 : 1)
            adapter.addNode(type: entry.id, x: 10, y: 10)
            XCTAssertEqual(owner.document.nodes.count, baseCount + 1)
            let inserted = try XCTUnwrap(owner.document.nodes.first { !addresses.contains($0.address) })
            let insertedID = try XCTUnwrap(adapter.projection.displayIDByAddress[inserted.address])
            adapter.selectNode(insertedID, extending: false)
            adapter.deleteSelection(); XCTAssertEqual(owner.document.nodes.count, baseCount)
            if let displaced {
                XCTAssertTrue(adapter.insert(payload: displaced.payload, address: displaced.address, name: displaced.name,
                                             x: displaced.x, y: displaced.y, lease: owner.lease()))
            }
            XCTAssertEqual(owner.document.nodes.count, count)
        }
        // Exercise the actual installed camera bridge with deterministic native events.
        // These are routed fixtures, not physical input-to-paint measurements.
        func camera(in view:NSView)->CanvasScrollController? {
            if let probe=view as? CanvasScrollProbe,let controller=probe.controller {return controller}
            for child in view.subviews {if let controller=camera(in:child) {return controller}}
            return nil
        }
        let controller=try XCTUnwrap(camera(in:host))
        let clip=try XCTUnwrap(controller.scrollView?.contentView)
        let center=clip.convert(CGPoint(x:clip.bounds.midX,y:clip.bounds.midY),to:nil)
        let initialOrigin=try XCTUnwrap(controller.origin)
        let unchanged=owner.document
        func mouse(_ type:NSEvent.EventType,_ index:Int)->NSEvent {
            NSEvent.mouseEvent(with:type,location:CGPoint(x:center.x+Double(index)*3,y:center.y+Double(index)*2),
                modifierFlags:[],timestamp:0,windowNumber:window.windowNumber,context:nil,eventNumber:index,
                clickCount:1,pressure:1)!
        }
        profiler.scenario = .panDrag
        XCTAssertNil(controller.input.routedEvent(mouse(.rightMouseDown,0)))
        try await run(.panDrag) { i in
            XCTAssertNil(controller.input.routedEvent(mouse(.rightMouseDragged,i+1)))
            XCTAssertEqual(controller.origin?.x,initialOrigin.x-Double(i+1)*3)
            XCTAssertEqual(controller.origin?.y,initialOrigin.y+Double(i+1)*2)
        }
        XCTAssertNil(controller.input.routedEvent(mouse(.rightMouseUp,iterations)))
        XCTAssertEqual(owner.document,unchanged)
        try await run(.wheelZoom) { i in
            let before=adapter.canvasViewState.zoom
            XCTAssertTrue(controller.input.routeWheel(point:CGPoint(x:500,y:350),inside:true,deltaX:0,
                deltaY:i % 2 == 0 ? 2:-2,precise:true,momentum:false))
            XCTAssertNotEqual(adapter.canvasViewState.zoom,before)
        }
        XCTAssertEqual(owner.document,unchanged)
        // Camera restore goes through the actual hosted canvas; this is not wheel input.
        try await run(.panZoom) { i in
            adapter.canvasViewState.zoom = i % 2 == 0 ? 0.65 : 0.7
            adapter.canvasViewState.scrollX = Double(i * 10)
            adapter.canvasViewState.scrollY = Double(i * 5)
            adapter.canvasViewRevision = UUID()
        }
        profiler.scenario = .workerBurst
        let finished = expectation(description: "Synthetic pipe burst drains before exit")
        var received = 0
        let burst = String(repeating: "{\"event\":\"profile-fixture\",\"epoch\":1}\n", count: 512)
        let process = try ProcessPump.launch(executable: URL(fileURLWithPath: "/usr/bin/printf"), arguments: ["%s", burst], onLine: { line in
            // Synthetic JSON workload, not a claim of training/protocol verification.
            _ = try? JSONSerialization.jsonObject(with: Data(line.utf8)); received += 1
        }, onExit: { code, _ in XCTAssertEqual(code, 0); finished.fulfill() })
        defer { if process.isRunning { process.terminate() } }
        try await run(.workerBurst) { i in adapter.selectNode(i % 2 == 0 ? first : second, extending: false) }
        await fulfillment(of: [finished], timeout: 10)
        XCTAssertEqual(received, 512)
        probe.stop()
        let report = profiler.stop()
        XCTAssertEqual(report.droppedSamples, 0)
        XCTAssertTrue(report.samples.contains { $0.metric == .canvasBody })
        XCTAssertTrue(report.samples.contains { $0.metric == .mainQueueDelay })
        let environment: [String: Any] = ["viewport": [1000, 700], "backingScale": window.backingScaleFactor,
            "appearance": NSApp.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua])?.rawValue ?? "unknown"]
        let captureURL = URL(fileURLWithPath: output)
        try JSONSerialization.data(withJSONObject: environment, options: [.sortedKeys]).write(to: captureURL.deletingLastPathComponent().appendingPathComponent("fixture.json"), options: .withoutOverwriting)
        if let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) {
            host.cacheDisplay(in: host.bounds, to: bitmap)
            if let png = bitmap.representation(using: .png, properties: [:]) {
                try png.write(to: captureURL.deletingLastPathComponent().appendingPathComponent("fixture.png"), options: .withoutOverwriting)
            }
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(report).write(to: URL(fileURLWithPath: output), options: .withoutOverwriting)
    }
}
