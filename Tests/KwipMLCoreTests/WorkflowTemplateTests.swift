import XCTest
@testable import KwipMLCore

final class WorkflowTemplateTests: XCTestCase {
    func testEveryTemplateCreatesIndependentPortableSourceWithoutPermissions() throws {
        let entries = try WorkflowTemplates.catalog()
        XCTAssertEqual(entries.count, 9)
        XCTAssertEqual(Set(entries.map(\.id)).count, entries.count)
        for entry in entries {
            let first = try WorkflowTemplates.instantiate(id: entry.id)
            var second = try WorkflowTemplates.instantiate(id: entry.id)
            XCTAssertNotEqual(first.projectID, second.projectID)
            XCTAssertEqual(first.nodes, second.nodes)
            XCTAssertEqual(first.edges, second.edges)
            XCTAssertEqual(first.templateOrigin?.id, entry.id)
            XCTAssertEqual(first.revision, 0)
            XCTAssertFalse(first.nodes.isEmpty)
            XCTAssertTrue(first.drafts.isEmpty)
            let data = try WorkflowCodec.encode(first)
            XCTAssertEqual(try WorkflowCodec.decode(data), first)
            let text = try XCTUnwrap(String(data: data, encoding: .utf8))
            for privateValue in ["/Users/", ".local/", "bookmark", "Bearer ", "127.0.0.1", "Showcase demos"] {
                XCTAssertFalse(text.contains(privateValue), "Template must not carry private authority: \(entry.id)")
            }
            second.name = "My independent workflow"
            XCTAssertNotEqual(second.name, first.name)
        }
        XCTAssertThrowsError(try WorkflowTemplates.instantiate(id: "../private"))
        XCTAssertThrowsError(try WorkflowTemplates.instantiate(id: "sales-reconciliation-pending"))
    }

    func testOldProjectsDoNotAcquireTemplateMetadata() throws {
        let old = WorkflowDocument(projectID: UUID().uuidString)
        let bytes = try WorkflowCodec.encode(old)
        XCTAssertNil(try WorkflowCodec.decode(bytes).templateOrigin)
        XCTAssertFalse(String(decoding: bytes, as: UTF8.self).contains("templateOrigin"))
    }

    func testItemSalesTemplateDerivesTwoMonthsAndRequiresNoPrivateInputs() throws {
        let document=try WorkflowTemplates.instantiate(id:"item-sales-reconciliation")
        XCTAssertEqual(document.nodes.count,81)
        XCTAssertEqual(document.edges.count,96)
        let source=try XCTUnwrap(document.nodes.first {$0.address.localID=="source"})
        guard case .dataOperator("kwip.data.sourceCSV",let config)=source.payload,
              case .object(let fields)=config["schemaOverrides"] else {return XCTFail("Explicit source fields are required")}
        XCTAssertEqual(Set(fields.keys),Set(["SourceRowID","InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID","Country"]))
        XCTAssertTrue(fields.values.allSatisfy {$0 == .string("String")},"Keep original lexical values; calculations own their conversions")
        let period=try XCTUnwrap(document.nodes.first {$0.address.localID=="require-two-months"})
        guard case .dataOperator("kwip.data.formula",let periodConfig)=period.payload,
              case .array(let operations)=periodConfig["operations"] else {return XCTFail("Missing two-month guard")}
        XCTAssertTrue(operations.contains {value in
            guard case .object(let operation)=value else {return false}
            return operation["expression"] == .string("require([MonthCount] = 2, [MinimumMonth])")
        })
        let portable=String(decoding:try WorkflowCodec.encode(document),as:UTF8.self)
        XCTAssertFalse(portable.contains("2011-"),"Template months come from the selected data, not recorded demo dates")
        let exports=document.nodes.compactMap {node -> [String:JSONValue]? in
            guard case .dataOperator("kwip.data.outputDelimited",let config)=node.payload else {return nil};return config
        }
        XCTAssertEqual(exports.count,3)
        XCTAssertTrue(exports.allSatisfy {$0["overwrite"] == .bool(false) && $0["destination"] == .string("file")})
        XCTAssertEqual(Set(exports.compactMap {$0["filename"]?.stringValue}).count,3)
    }

    func testTemplateMetadataRequiresItsVersionedSchema() throws {
        var document = try WorkflowTemplates.instantiate(id: "sales-exceptions")
        document.schemaVersion = 23
        XCTAssertThrowsError(try WorkflowCodec.validate(document))
        document.schemaVersion = 24
        document.templateOrigin = .init(id: "../bad", version: 1)
        XCTAssertThrowsError(try WorkflowCodec.validate(document))
    }
}
