import Foundation
import VisionCamera
import MLKitVision
import MLKitTextRecognition
import MLKitTextRecognitionChinese
import MLKitTextRecognitionDevanagari
import MLKitTextRecognitionJapanese
import MLKitTextRecognitionKorean
import MLKitCommon
import CoreImage

@objc(VisionCameraTextRecognition)
public class VisionCameraTextRecognition: FrameProcessorPlugin {

    private var textRecognizer = TextRecognizer()
    private static let latinOptions = TextRecognizerOptions()
    private static let chineseOptions = ChineseTextRecognizerOptions()
    private static let devanagariOptions = DevanagariTextRecognizerOptions()
    private static let japaneseOptions = JapaneseTextRecognizerOptions()
    private static let koreanOptions = KoreanTextRecognizerOptions()
    private var data: [String: Any] = [:]

    // Used when converting the frame pixel format.
    private let context = CIContext(options: nil)
    private var bufferPool: CVPixelBufferPool?


    public override init(proxy: VisionCameraProxyHolder, options: [AnyHashable: Any]! = [:]) {
        super.init(proxy: proxy, options: options)
        let language = options["language"] as? String ?? "latin"
        switch language {
        case "chinese":
            self.textRecognizer = TextRecognizer.textRecognizer(options: VisionCameraTextRecognition.chineseOptions)
        case "devanagari":
            self.textRecognizer = TextRecognizer.textRecognizer(options: VisionCameraTextRecognition.devanagariOptions)
        case "japanese":
            self.textRecognizer = TextRecognizer.textRecognizer(options: VisionCameraTextRecognition.japaneseOptions)
        case "korean":
            self.textRecognizer = TextRecognizer.textRecognizer(options: VisionCameraTextRecognition.koreanOptions)
        default:
            self.textRecognizer = TextRecognizer.textRecognizer(options: VisionCameraTextRecognition.latinOptions)
        }
    }


    public override func callback(_ frame: Frame, withArguments arguments: [AnyHashable: Any]?) -> Any {
        let buffer = frame.buffer
        let pixelFormat = frame.pixelFormat
        let image: VisionImage

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(buffer) else {
            print("Failed to get CVPixelBuffer from frame.")
            return [:]
       }

        let pixelFormatType = CVPixelBufferGetPixelFormatType(pixelBuffer)

        // These are the accepted MLKit pixel formats.
        let compatibleFormats: [OSType] = [
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            kCVPixelFormatType_32BGRA
        ]

        // Make sure the frame pixel format is compatible.
        if compatibleFormats.contains(pixelFormatType) {
            image = VisionImage(buffer: buffer)
        } else {
            guard let bgraPixelBuffer = convertToBGRA(from: pixelBuffer) else {
                print("Failed to convert pixel buffer to BGRA.")
                return [:]
            }

            guard let newSampleBuffer = createSampleBuffer(from: bgraPixelBuffer, withTimingFrom: frame.buffer) else {
                print("Failed to create new sample buffer.")
                return [:]
            }

            image = VisionImage(buffer: newSampleBuffer)
        }

        image.orientation = getOrientation(orientation: frame.orientation)

        do {
            let result = try self.textRecognizer.results(in: image)
            let blocks = VisionCameraTextRecognition.processBlocks(blocks: result.blocks)
            data["resultText"] = result.text
            data["blocks"] = blocks
            if result.text.isEmpty {
                return [:]
            }else{
                return data
            }
        } catch {
            print("Failed to recognize text: \(error.localizedDescription).")
            return [:]
        }
    }

    /**
     * Converts a CVPixelBuffer from an unknown YUV format to kCVPixelFormatType_32BGRA.
     */
    private func convertToBGRA(from pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        // Create a CIImage from the source pixel buffer
        let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)

        // On the first run, create a buffer pool with the correct BGRA format.
        // The pool is expensive to create, so we only do it once.
        if bufferPool == nil {
            let poolAttributes = [kCVPixelBufferPoolMinimumBufferCountKey: 3] as CFDictionary
            let bufferAttributes = [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey: CVPixelBufferGetWidth(pixelBuffer),
                kCVPixelBufferHeightKey: CVPixelBufferGetHeight(pixelBuffer),
                kCVPixelFormatOpenGLESCompatibility: true,
                kCVPixelBufferIOSurfacePropertiesKey: [:]
            ] as CFDictionary
            let status = CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttributes, bufferAttributes, &bufferPool)
            if status != kCVReturnSuccess {
                print("Failed to create CVPixelBufferPool. Status: \(status)")
                return nil
            }
        }

        // Pull a new, empty CVPixelBuffer from our pool. This is much faster than allocating a new one.
        var convertedPixelBuffer: CVPixelBuffer?
        guard let pool = bufferPool,
              CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &convertedPixelBuffer) == kCVReturnSuccess,
              let finalBuffer = convertedPixelBuffer else {
            print("Failed to create pixel buffer from pool.")
            return nil
        }

        // Render the CIImage into our new, empty, BGRA-formatted CVPixelBuffer.
        // This render operation is executed on the GPU and is extremely fast.
        self.context.render(sourceImage, to: finalBuffer)

        return finalBuffer
    }

     /**
     * Creates a new CMSampleBuffer around a CVPixelBuffer.
     * It copies the timing information from the original sample buffer.
     */
    private func createSampleBuffer(from pixelBuffer: CVPixelBuffer, withTimingFrom originalBuffer: CMSampleBuffer) -> CMSampleBuffer? {
        var timingInfo = CMSampleTimingInfo()
        guard CMSampleBufferGetSampleTimingInfo(originalBuffer, at: 0, timingInfoOut: &timingInfo) == noErr else {
            print("Failed to get timing info from original buffer.")
            return nil
        }

        var videoInfo: CMVideoFormatDescription?
        guard CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                                           imageBuffer: pixelBuffer,
                                                           formatDescriptionOut: &videoInfo) == noErr else {
            print("Failed to create video format description.")
            return nil
        }

        do {
            let sampleBuffer = try CMSampleBuffer(imageBuffer: pixelBuffer,
                                                  formatDescription: videoInfo!,
                                                  sampleTiming: timingInfo)
            return sampleBuffer
        } catch {
            print("Failed to create sample buffer from image buffer: \(error.localizedDescription)")
            return nil
        }
    }



    static func processBlocks(blocks:[TextBlock]) -> Array<Any> {
        var blocksArray : [Any] = []
        for block in blocks {
            var blockData : [String:Any] = [:]
            blockData["blockText"] = block.text
            blockData["blockCornerPoints"] = processCornerPoints(block.cornerPoints)
            blockData["blockFrame"] = processFrame(block.frame)
            blockData["lines"] = processLines(lines: block.lines)
            blocksArray.append(blockData)
        }
        return blocksArray
    }

    private static func processLines(lines:[TextLine]) -> Array<Any> {
        var linesArray : [Any] = []
        for line in lines {
            var lineData : [String:Any] = [:]
            lineData["lineText"] = line.text
            lineData["lineLanguages"] = processRecognizedLanguages(line.recognizedLanguages)
            lineData["lineCornerPoints"] = processCornerPoints(line.cornerPoints)
            lineData["lineFrame"] = processFrame(line.frame)
            lineData["elements"] = processElements(elements: line.elements)
            linesArray.append(lineData)
        }
        return linesArray
    }

    private static func processElements(elements:[TextElement]) -> Array<Any> {
        var elementsArray : [Any] = []

        for element in elements {
            var elementData : [String:Any] = [:]
              elementData["elementText"] = element.text
              elementData["elementCornerPoints"] = processCornerPoints(element.cornerPoints)
              elementData["elementFrame"] = processFrame(element.frame)

            elementsArray.append(elementData)
          }

        return elementsArray
    }

    private static func processRecognizedLanguages(_ languages: [TextRecognizedLanguage]) -> [String] {

            var languageArray: [String] = []

            for language in languages {
                guard let code = language.languageCode else {
                    print("No language code exists")
                    break;
                }
                if code.isEmpty{
                    languageArray.append("und")
                }else {
                    languageArray.append(code)

                }
            }

            return languageArray
        }

    private static func processCornerPoints(_ cornerPoints: [NSValue]) -> [[String: CGFloat]] {
        return cornerPoints.compactMap { $0.cgPointValue }.map { ["x": $0.x, "y": $0.y] }
    }

    private static func processFrame(_ frameRect: CGRect) -> [String: CGFloat] {
        let offsetX = (frameRect.midX - ceil(frameRect.width)) / 2.0
        let offsetY = (frameRect.midY - ceil(frameRect.height)) / 2.0

        let x = frameRect.maxX + offsetX
        let y = frameRect.minY + offsetY

        return [
            "x": frameRect.midX + (frameRect.midX - x),
            "y": frameRect.midY + (y - frameRect.midY),
            "width": frameRect.width,
            "height": frameRect.height,
            "boundingCenterX": frameRect.midX,
            "boundingCenterY": frameRect.midY
    ]
    }

    private func getOrientation(orientation: UIImage.Orientation) -> UIImage.Orientation {
        switch orientation {
        case .up:
          return .up
        case .left:
          return .right
        case .down:
          return .down
        case .right:
          return .left
        default:
          return .up
        }
    }
}
