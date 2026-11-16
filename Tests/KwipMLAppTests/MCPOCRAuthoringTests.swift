import XCTest
import AppKit
@testable import KwipMLCore
@testable import KwipMLApp

/// Actual case03 author request, stripped of project identity; no private grants or worker runs.
@MainActor final class MCPOCRAuthoringTests: XCTestCase {
    func testActualOCRAuthorRequestReportsInvalidEnumsAndNormalizedMatte() async throws {
        let operations = try JSONDecoder().decode([JSONValue].self, from: Data(Self.request.utf8))
        guard case .object(let sourceOperation)=operations[7], case .object(let source)?=sourceOperation["configuration"],
              case .object(let mediaOperation)=operations[8], case .object(let media)?=mediaOperation["configuration"] else { return XCTFail("Fixture shape") }
        XCTAssertThrowsError(try FileInputConfiguration.validate(source)) { print("CASE03_SOURCE_REFUSAL: \($0)") }
        XCTAssertThrowsError(try PrepareMediaConfiguration.validate(media)) { print("CASE03_MEDIA_REFUSAL: \($0)") }
        var correctedSource=source;correctedSource["unknownFiles"] = .string("error")
        var correctedMedia=media;correctedMedia["matte"] = .array([1,1,1])
        XCTAssertNoThrow(try FileInputConfiguration.validate(correctedSource))
        XCTAssertNoThrow(try PrepareMediaConfiguration.validate(correctedMedia))
        _=NSApplication.shared
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent("mcp-ocr-author-"+UUID().uuidString)
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let windows=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)})
        let window=try windows.newFlow(present:false)
        let owner=try XCTUnwrap(window.session.workspace.workflowSession?.owner)
        let controller=MCPController(directory:directory,secrets:MCPWorkflowGatewayTests.Secrets(),startConfigured:false)
        var settings=MCPSettings.factory;settings.disabledComponents.remove("kwip.data.outputDelimited");settings.enabled=true;settings.clients=[.init(id:"test",name:"Test",actions:Set(MCPAction.allCases))];settings.projects[owner.projectID] = .init(actions:Set(MCPAction.allCases),readCapturedValues:true)
        try controller.save(settings)
        defer {for w in windows.openControllers {_=windows.close(w)};controller.stop(persist:false);try? FileManager.default.removeItem(at:directory)}
        let gateway=MCPWorkflowGateway(controller:controller,windows:windows)
        // Discovery must publish the same contracts enforced by native authoring.
        let catalog=try await gateway.dispatch(clientID:"test",action:"catalog_list",arguments:[:])
        guard case .array(let entries)?=catalog["components"] else {return XCTFail("Missing discovery catalog")}
        func completeMetadata(arguments:[String:JSONValue],response:[String:JSONValue])async throws->JSONValue {
        guard case .object(var page)=response["metadataRead"] else {return .object(response)}
        let kind=page["kind"]?.stringValue
        if let value=page["value"],kind != "string" {return value}
        var fields:[String:JSONValue]=[:],items:[JSONValue]=[],text=""
        while true {
            if kind == "string" {text += page["value"]?.stringValue ?? ""}
            if case .array(let entries)=page["entries"] {
                for entry in entries {
                    guard case .object(let row)=entry,let key=row["key"]?.stringValue else {continue}
                    let value:JSONValue
                    if let complete=row["value"] {value=complete}
                    else {
                        var child=arguments;child["readPath"]=row["readPath"];child["readRevision"]=page["readRevision"]
                        value=try await completeMetadata(arguments:child,response:try await gateway.dispatch(clientID:"test",action:"catalog_get",arguments:child))
                    }
                    if kind == "object" {fields[key]=value} else {items.append(value)}
                }
            }
            guard case .object(let next)=page["nextRead"] else {break}
            let request=arguments.merging(next){_,new in new}
            let result=try await gateway.dispatch(clientID:"test",action:"catalog_get",arguments:request)
            guard case .object(let nextPage)=result["metadataRead"] else {throw MCPSettings.Refusal(code:"missing_page")};page=nextPage
        }
        if kind == "object" {return .object(fields)}
        if kind == "array" {return .array(items)}
        return .string(text)
    }
        func schema(_ family:String)async throws->JSONValue {
            let entry=try XCTUnwrap(entries.first { if case .object(let row)=$0 {return row["familyID"] == .string(family)};return false })
            guard case .object(let row)=entry else {throw NSError(domain:"fixture",code:1)}
            let args:[String:JSONValue] = ["componentID":try XCTUnwrap(row["componentID"])]
            let response=try await gateway.dispatch(clientID:"test",action:"catalog_get",arguments:args)
            let complete=try await completeMetadata(arguments:args,response:response)
            guard case .object(let detail)=complete,case .array(let items)=detail["components"],case .object(let item)=items.first else {throw NSError(domain:"fixture",code:2)}
            return try XCTUnwrap(item["configurationSchema"])
        }
        let sourceSchema=try await schema("kwip.data.sourceFiles"),mediaSchema=try await schema("kwip.data.prepareMedia")
        XCTAssertThrowsError(try WorkflowSchema.check(.object(source),schema:sourceSchema),"Discovery must reject unsupported unknownFiles=fail")
        XCTAssertThrowsError(try WorkflowSchema.check(.object(media),schema:mediaSchema),"Discovery must reject byte-range matte=255")
        for choice in ["metadata","error","skip"] {
            var value=correctedSource;value["unknownFiles"] = .string(choice)
            XCTAssertNoThrow(try WorkflowSchema.check(.object(value),schema:sourceSchema));XCTAssertNoThrow(try FileInputConfiguration.validate(value))
        }
        for channel:Double in [-0.01,0,0.5,1,1.01] {
            var value=correctedMedia;value["matte"] = .array([.number(channel),0,1])
            if (0...1).contains(channel) {
                XCTAssertNoThrow(try WorkflowSchema.check(.object(value),schema:mediaSchema));XCTAssertNoThrow(try PrepareMediaConfiguration.validate(value))
            } else {
                XCTAssertThrowsError(try WorkflowSchema.check(.object(value),schema:mediaSchema));XCTAssertThrowsError(try PrepareMediaConfiguration.validate(value))
            }
        }
        let original=owner.document
        let refused=try await gateway.dispatch(clientID:"test",action:"patch_preview",arguments:["projectID":.string(owner.projectID),"revision":0,"operations":.array(operations)])
        XCTAssertEqual(refused["isError"],.bool(true))
        XCTAssertEqual(refused["code"],.string("invalid_configuration"))
        XCTAssertNil(refused["proposalID"],"Invalid original request must not create a proposal")
        XCTAssertEqual(owner.document,original)
        var fixed=operations;var a=sourceOperation;a["configuration"] = .object(correctedSource);fixed[7] = .object(a);var b=mediaOperation;b["configuration"] = .object(correctedMedia);fixed[8] = .object(b)
        for length in 1...fixed.count {
            do {
                let prefix=try await gateway.dispatch(clientID:"test",action:"patch_preview",arguments:["projectID":.string(owner.projectID),"revision":0,"operations":.array(Array(fixed.prefix(length)))])
                XCTAssertNotNil(prefix["proposalID"],"Corrected prefix \(length) must create a proposal")
                XCTAssertNotEqual(prefix["isError"],.bool(true))
                XCTAssertEqual(owner.document,original)
                print("CASE03_PREFIX_OK: \(length)")
            }
            catch {print("CASE03_PREFIX_REFUSED: \(length) \(String(reflecting:type(of:error))) \(error)");throw error}
        }
        let preview=try await gateway.dispatch(clientID:"test",action:"patch_preview",arguments:["projectID":.string(owner.projectID),"revision":0,"operations":.array(fixed)])
        XCTAssertNotNil(preview["proposalID"]);XCTAssertEqual(owner.document,original)
        print("CASE03_CORRECTED_PREVIEW: admitted, no source mutation")
    }
    private static let request = #"""
[
  {
    "operation": "insert",
    "componentID": "wf-create-WyJkYXRhIiwia3dpcC5kYXRhLnNvdXJjZUZpbGVzIl0=",
    "nodeID": "source",
    "x": 0,
    "y": 0
  },
  {
    "operation": "insert",
    "componentID": "wf-create-WyJkYXRhIiwia3dpcC5kYXRhLnByZXBhcmVNZWRpYSJd",
    "nodeID": "pixels",
    "x": 300,
    "y": 0
  },
  {
    "operation": "insert",
    "componentID": "wf-create-WyJkYXRhIiwia3dpcC5kYXRhLmV4dHJhY3RUZXh0Il0=",
    "nodeID": "ocr",
    "x": 600,
    "y": 0
  },
  {
    "operation": "insert",
    "componentID": "wf-create-WyJkYXRhIiwia3dpcC5kYXRhLnZpZXciXQ==",
    "nodeID": "documents-view",
    "x": 900,
    "y": -150
  },
  {
    "operation": "insert",
    "componentID": "wf-create-WyJkYXRhIiwia3dpcC5kYXRhLnZpZXciXQ==",
    "nodeID": "regions-view",
    "x": 900,
    "y": 150
  },
  {
    "operation": "insert",
    "componentID": "wf-create-WyJkYXRhIiwia3dpcC5kYXRhLm91dHB1dERlbGltaXRlZCJd",
    "nodeID": "documents-csv",
    "x": 1200,
    "y": -150
  },
  {
    "operation": "insert",
    "componentID": "wf-create-WyJkYXRhIiwia3dpcC5kYXRhLm91dHB1dERlbGltaXRlZCJd",
    "nodeID": "regions-csv",
    "x": 1200,
    "y": 150
  },
  {
    "operation": "configure",
    "address": {
      "domain": "data",
      "localID": "source"
    },
    "configuration": {
      "errorPolicy": "fail",
      "excludePatterns": [],
      "extensions": [
        "png"
      ],
      "includeHidden": false,
      "includePatterns": [
        "**/*"
      ],
      "includeSubfolders": false,
      "maxDepth": 32,
      "maxFileBytes": 16777216,
      "maxFiles": 3,
      "maxTotalBytes": 50331648,
      "metadataFields": [
        "file_id",
        "name",
        "relative_path",
        "extension",
        "detected_type",
        "byte_size"
      ],
      "readMode": "content",
      "readers": {
        "docx": {
          "include_footers": false,
          "include_headers": false,
          "include_nested_tables": true,
          "include_tables": true,
          "separator": "\n\n",
          "units": "file"
        },
        "image": {
          "colorMode": "preserve",
          "frameEnd": 0,
          "frameStart": 0,
          "frames": "first",
          "layout": "HWC",
          "orientation": "stored",
          "palette": "expandRGBA"
        },
        "odt": {
          "include_tables": true,
          "separator": "\n\n",
          "units": "file"
        },
        "pdf": {
          "extraction_mode": "plain",
          "layout_font_height_weight": 1,
          "layout_scale_weight": 1.25,
          "layout_space_vertically": true,
          "layout_strip_rotated": true,
          "maximumRenderPages": 20,
          "orientations": [
            0,
            90,
            180,
            270
          ],
          "pages": [],
          "renderDPI": 144,
          "renderMode": "text",
          "separator": "\n\n",
          "space_width": 200,
          "strict": false,
          "units": "file"
        },
        "rtf": {
          "encoding": "cp1252",
          "errors": "strict",
          "separator": "\n\n",
          "units": "file"
        },
        "text": {
          "encoding": "auto",
          "errors": "strict",
          "keep_empty_lines": true,
          "newline": "preserve",
          "units": "file"
        }
      },
      "sourceMode": "directory",
      "symlinks": "skip",
      "unknownFiles": "fail"
    }
  },
  {
    "operation": "configure",
    "address": {
      "domain": "data",
      "localID": "pixels"
    },
    "configuration": {
      "alpha": "composite",
      "center": true,
      "channelIndex": 0,
      "channels": "preserve",
      "color": "RGB",
      "cropEnabled": false,
      "cropHeight": 224,
      "cropWidth": 224,
      "cropX": 0,
      "cropY": 0,
      "fftSize": 400,
      "filterRadius": 16,
      "height": 1025,
      "hopSamples": 0,
      "inputColumn": "image_content",
      "interpolation": "bilinear",
      "kind": "image",
      "layout": "HWC",
      "logFloorDB": -80,
      "maskConvention": "valid",
      "matte": [
        255,
        255,
        255
      ],
      "maximumHz": 0,
      "melBands": 80,
      "minimumHz": 0,
      "outputColumn": "ocr_image",
      "outputMode": "image",
      "representation": "samples",
      "rotation": 0,
      "sampleFormat": "preserve",
      "scaling": "none",
      "sizeMode": "preserve",
      "spectralHop": 160,
      "tail": "drop",
      "targetRate": 0,
      "width": 1025,
      "window": "hann",
      "windowSamples": 0
    }
  },
  {
    "operation": "configure",
    "address": {
      "domain": "data",
      "localID": "ocr"
    },
    "configuration": {
      "allowlist": "",
      "beamWidth": 5,
      "blocklist": "",
      "canvasSize": 1100,
      "decoder": "greedy",
      "device": "cpu",
      "groupBy": "file_id",
      "heightThreshold": 0.5,
      "inputColumn": "ocr_image",
      "linkThreshold": 0.4,
      "lowText": 0.4,
      "magnification": 1,
      "margin": 0.1,
      "maximumCharacters": 100000,
      "maximumRegions": 1000,
      "minimumSize": 10,
      "outputColumn": "document_text",
      "pack": "english-printed-v1",
      "pageColumn": "frame_index",
      "reviewThreshold": 0.6,
      "rotationCandidates": [],
      "separator": "\n\n",
      "slopeThreshold": 0.1,
      "textThreshold": 0.7,
      "widthThreshold": 0.5,
      "ycenterThreshold": 0.5
    }
  },
  {
    "operation": "configure",
    "address": {
      "domain": "data",
      "localID": "documents-csv"
    },
    "configuration": {
      "columns": [],
      "filename": "documents-a.csv",
      "overwrite": false,
      "separator": ",",
      "format": "csv",
      "destination": "file"
    }
  },
  {
    "operation": "configure",
    "address": {
      "domain": "data",
      "localID": "regions-csv"
    },
    "configuration": {
      "columns": [],
      "filename": "regions-a.csv",
      "overwrite": false,
      "separator": ",",
      "format": "csv",
      "destination": "file"
    }
  }
]
"""#
}
