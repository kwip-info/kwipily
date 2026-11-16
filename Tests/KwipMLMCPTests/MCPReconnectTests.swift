import XCTest
@testable import KwipMLMCP

/// Real sockets and SDK lifecycle, not a mocked initialize handler.
final class MCPReconnectTests: XCTestCase {
    private let token = String(repeating: "r", count: 43)
    private func start() async throws -> (MCPLocalHTTPBroker, Int) {
        let port = Int.random(in: 30000...50000)
        let broker = try MCPLocalHTTPBroker(setup: .init(fixedPort: port, clients: [
            .init(id: "demo", tokenSHA256: MCPHTTPAdmission.tokenDigest(token))
        ])) { client, action, arguments in
            if action == "tools_list" {
                return Data(#"{"tools":[{"name":"echo","description":"Echo synthetic input","inputSchema":{"type":"object"}}]}"#.utf8)
            }
            let value = try JSONSerialization.jsonObject(with: arguments) as! [String: Any]
            if value["slow"] as? Bool == true { try await Task.sleep(nanoseconds: 200_000_000) }
            return try JSONSerialization.data(withJSONObject: ["client": client, "value": value["value"] ?? ""])
        }
        let ready = expectation(description: "listener ready")
        broker.start { actual, error in
            XCTAssertEqual(actual, UInt16(port)); XCTAssertNil(error); ready.fulfill()
        }
        await fulfillment(of: [ready], timeout: 5)
        return (broker, port)
    }
    private func call(_ port: Int, _ method: String, _ params: [String: Any] = [:], token: String? = nil) async throws -> (Int, [String: Any]) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/mcp")!)
        request.httpMethod = "POST"; request.timeoutInterval = 5
        request.setValue("Bearer " + (token ?? self.token), forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        // Independent harnesses legitimately reuse JSON-RPC request IDs.
        request.httpBody = try JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "id": 1, "method": method, "params": params])
        let (data, response) = try await URLSession.shared.data(for: request)
        return ((response as! HTTPURLResponse).statusCode, data.isEmpty ? [:] : try JSONSerialization.jsonObject(with: data) as! [String: Any])
    }
    private var initialization: [String: Any] {
        ["protocolVersion": "2025-03-26", "capabilities": [:], "clientInfo": ["name": "independent-client", "version": "1"]]
    }
    func testIndependentClientsCanInitializeAndReconnectWithSameCredential() async throws {
        let (broker, port) = try await start(); defer { broker.stop() }
        for _ in 0..<3 {
            let (status, value) = try await call(port, "initialize", initialization)
            XCTAssertEqual(status, 200); XCTAssertNil(value["error"], "\(value)")
            XCTAssertNotNil((value["result"] as? [String: Any])?["protocolVersion"])
            let (_, tools) = try await call(port, "tools/list")
            XCTAssertNotNil((tools["result"] as? [String: Any])?["tools"])
        }
    }
    func testConcurrentIdenticalRequestIDsDoNotCrossResponses() async throws {
        let (broker, port) = try await start(); defer { broker.stop() }
        _ = try await call(port, "initialize", initialization)
        async let slow = call(port, "tools/call", ["name": "echo", "arguments": ["value": "slow", "slow": true]])
        async let fast = call(port, "tools/call", ["name": "echo", "arguments": ["value": "fast"]])
        let responses = try await [slow, fast]
        for (response, expected) in zip(responses, ["slow", "fast"]) {
            XCTAssertEqual(response.0, 200)
            XCTAssertNil(response.1["error"])
            let result = response.1["result"] as? [String: Any]
            XCTAssertEqual((result?["structuredContent"] as? [String: Any])?["value"] as? String, expected)
            let content = result?["content"] as? [[String:Any]]
            XCTAssertEqual(content?.first?["text"] as? String,"Result available in structuredContent.")
        }
    }
    func testCredentialStillRequiredAfterAnotherClientInitialized() async throws {
        let (broker, port) = try await start(); defer { broker.stop() }
        _ = try await call(port, "initialize", initialization)
        let (status, _) = try await call(port, "tools/list", token: String(repeating: "x", count: 43))
        XCTAssertEqual(status, 401)
    }
}
