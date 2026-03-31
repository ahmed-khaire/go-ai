package xai

import "github.com/digitallysavvy/go-ai/pkg/provider/types"

// ViewXVideo creates a provider-executed tool that analyzes and describes an X (Twitter) video.
// Processing is handled by xAI's servers.
//
// Example:
//
//	tool := xai.ViewXVideo()
func ViewXVideo() types.Tool {
	return types.Tool{
		Name:             "xai.view_x_video",
		Description:      "Analyze and describe an X (Twitter) video.",
		ProviderExecuted: true,
		Execute:          providerExecutedNoop("xai.view_x_video"),
	}
}
