import XCTest
import MCP
@testable import KwipMLMCP
final class HTTPAdmissionTests:XCTestCase {
    let token=String(repeating:"a",count:43)
    func testLocalAuthenticatedRequestAndRevocation()throws {
        let c=MCPBrokerClient(id:"synthetic",tokenSHA256:MCPHTTPAdmission.tokenDigest(token))
        let r=HTTPRequest(method:"POST",headers:["Host":"127.0.0.1:8788","Authorization":"Bearer "+token],body:Data(),path:"/mcp")
        XCTAssertEqual(try MCPHTTPAdmission.client(for:r,port:8788,clients:[c]),"synthetic")
        XCTAssertThrowsError(try MCPHTTPAdmission.client(for:r,port:8788,clients:[]))
        for host in ["evil.example:8788","127.0.0.1:8789","127.0.0.1","localhost:8788.evil"]{
            let other=HTTPRequest(method:"POST",headers:["Host":host,"Authorization":"Bearer "+token],path:"/mcp")
            XCTAssertThrowsError(try MCPHTTPAdmission.client(for:other,port:8788,clients:[c]))
        }
    }
    func testForeignOriginAndWrongTokenRefused()throws {
        let c=MCPBrokerClient(id:"synthetic",tokenSHA256:MCPHTTPAdmission.tokenDigest(token))
        for extra in [["Origin":"https://evil.example"],["Authorization":"Bearer "+String(repeating:"b",count:43)]]{
            let r=HTTPRequest(method:"POST",headers:["Host":"127.0.0.1:8788","Authorization":"Bearer "+token].merging(extra){_,v in v},path:"/mcp")
            XCTAssertThrowsError(try MCPHTTPAdmission.client(for:r,port:8788,clients:[c]))
        }
    }
    func testPartialFramingAndAmbiguity()throws {
        var p=MCPHTTPParser()
        XCTAssertNil(try p.append(Data("POST /mcp HTTP/1.1\r\nHost: localhost:8788\r\nContent-Length: 2\r\n\r\n{".utf8)))
        XCTAssertEqual(try p.append(Data("}".utf8))?.body,Data("{}".utf8))
        for header in ["Content-Length: 2\r\ncontent-length: 2","Transfer-Encoding: chunked\r\nContent-Length: 2","Content-Length: -1","Content-Length: 02","Content-Length: 1048577","Host: a\r\nHost: b"]{
            var p=MCPHTTPParser();XCTAssertThrowsError(try p.append(Data("POST /mcp HTTP/1.1\r\n\(header)\r\n\r\n{}".utf8)))
        }
    }
    func testHeadersAndPipelinedBytesBounded()throws {
        var p=MCPHTTPParser();XCTAssertThrowsError(try p.append(Data(repeating:65,count:16385)))
        p=MCPHTTPParser();XCTAssertThrowsError(try p.append(Data("POST /mcp HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}EXTRA".utf8)))
    }
}
